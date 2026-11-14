"""Gait specifications: the single place that defines what each gait is.

Everything gait-specific lives on a GaitSpec: speed envelope, foot-contact clock,
stride frequency vs speed, spawn behaviour, terminations, SGS binning and the stage
budgets. Env configs, rewards, the pipeline and the handoff recorder all read from here,
so adding a gait (e.g. a pace or a rotary gallop) = adding one GaitSpec to GAITS and a
reward module in rewards/.

Leg order everywhere in this file is LEG_NAMES = (FR, FL, RR, RL).

Clock convention: every leg runs the same cycle. A leg is in stance while
    ((cycles - touchdown[leg]) mod 1) < duty_factor
where `cycles` counts gait cycles (integrated stride frequency). touchdown[leg] is the
cycle fraction at which that foot lands, which is how gait diagrams are usually drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from centaur_rl.assets.robots.centaur.centaur_constants import LEG_NAMES

assert LEG_NAMES == ("FR", "FL", "RR", "RL"), "touchdown tables below assume this order"

# Leg length (hip to foot in the default stance); used only for Froude-number notes.
LEG_LENGTH = 0.58

# Physics timestep (s). The control step is GaitSpec.decimation of these.
SIM_DT = 0.005
# Control step the reward weights were first written for (50 Hz). Sparse once-per-event
# terms are paid weight * control_dt per event (mjlab scales rewards by dt), so a gait at
# another rate multiplies their weights by REFERENCE_DT / control_dt to keep them the same.
REFERENCE_DT = 0.02


def froude(v: float) -> float:
  return v * v / (9.81 * LEG_LENGTH)


@dataclass(frozen=True)
class GaitClock:
  """Foot-contact pattern of a gait, in LEG_NAMES order (FR, FL, RR, RL)."""

  touchdown: tuple[float, float, float, float]
  """Cycle fraction at which each foot lands (stance starts)."""
  duty_factor: float
  """Fraction of the cycle each foot is on the ground. < 0.5 for every leg => flight.
  With `air_time` set this is only the nominal value (used for the first guess)."""
  air_time: float | tuple[float, float, float, float] | None = None
  """If set, the time each foot spends in the air is held at this many seconds at every
  speed and the duty factor follows the stride frequency: duty = 1 - air_time * f,
  clamped to [duty_min, duty_max]. This is what horses do (Robilliard et al. 2007: swing
  time hardly changes with speed, stance time shrinks). A 4-tuple (LEG_NAMES order) gives
  each leg its own air time, hence its own duty factor (the gallop: hind stance is longer
  than fore stance, Witte et al. 2006)."""
  duty_min: float = 0.0
  duty_max: float = 1.0

  @property
  def leg_air_times(self) -> tuple[float, float, float, float] | None:
    if self.air_time is None or isinstance(self.air_time, tuple):
      return self.air_time
    return (self.air_time,) * 4

  def leg_duties_at(self, frequency: float) -> tuple[float, ...]:
    """Per-leg duty factor (LEG_NAMES order) at this stride frequency."""
    air = self.leg_air_times
    if air is None:
      return (self.duty_factor,) * 4
    return tuple(min(max(1.0 - a * frequency, self.duty_min), self.duty_max) for a in air)

  def duty_at(self, frequency: float) -> float:
    """Mean duty over the legs (= every leg's duty unless air_time is per leg). This is
    what the 1-wide `gait_duty` observation shows."""
    d = self.leg_duties_at(frequency)
    return sum(d) / len(d)

  @property
  def offsets(self) -> tuple[float, ...]:
    """Per-leg phase offset: stance while ((cycles + offset) mod 1) < duty_factor."""
    return tuple((-t) % 1.0 for t in self.touchdown)


@dataclass(frozen=True)
class StrideFrequency:
  """Stride frequency as a function of commanded speed.

  f(v) = clamp(v * duty / stance_sweep, f_min, f_max): a planted foot sweeps
  v * duty / f metres under its hip, which is held at stance_sweep until a frequency
  limit is reached; past that the stride lengthens instead. So frequency rises with
  speed inside a gait, as in real horses, instead of the stride stretching alone.
  """

  stance_sweep: float  # m of foot travel per stance
  f_min: float  # Hz
  f_max: float  # Hz
  slope: float | None = None
  """If set, f(v) = clamp(offset + slope * v, f_min, f_max) instead (the walk: strides
  lengthen faster than they quicken, as in Robilliard et al. 2007)."""
  offset: float = 0.0

  def __call__(self, v: float, duty: float) -> float:
    if self.slope is not None:
      return min(max(self.offset + self.slope * v, self.f_min), self.f_max)
    return min(max(v * duty / self.stance_sweep, self.f_min), self.f_max)


@dataclass(frozen=True)
class StageBudget:
  """Iteration budgets for one gait's stages (see pipeline.py)."""

  ladder_iterations: int = 1000
  """SGS speed ladder (V_current climbs from v_min to v_max)."""
  fork_iteration: int | None = None
  """Checkpoint of THIS gait's ladder that the NEXT gait forks from. None = last one."""
  refine_iterations: int = 10000
  """Whole-range refinement after the ladder. Produces the gait's policy."""
  handoff_iterations: int = 500
  """Optional handoff fine-tune on neighbouring gaits' states. 0 disables the stage."""
  critic_warmup_iterations: int = 200
  """Fork only: iterations at the start of the ladder with the actor frozen."""
  fork_std_floor: float = 0.6
  """Fork only: action std is raised to at least this so the new gait can explore."""


@dataclass(frozen=True)
class GaitSpec:
  name: str
  index: int  # position in the gait one-hot (fixed forever: changes the obs layout)
  speed_range: tuple[float, float]  # m/s, the gait's commanded forward-speed envelope
  clock: GaitClock
  stride: StrideFrequency

  # SGS speed bins over [max(speed_range[0], sgs_min_speed), speed_range[1]].
  speed_bins: int = 10
  sgs_min_speed: float = 0.0

  # Spawn. Gaits other than the walk never start from rest: they spawn already moving
  # at the episode's commanded speed (they never have to learn to accelerate from 0).
  spawn_from_rest: bool = False
  spawn_joint_noise: float = 0.05  # rad, uniform noise on the default pose
  random_start_phase: bool = True  # clock starts at a random cycle fraction

  # Handoff states (recorded from neighbouring gaits' policies, see tools/).
  refine_handoff_prob: float = 0.2  # share of refine resets started from a bank
  handoff_stage_prob: float = 0.5  # share of handoff-stage resets started from a bank

  # Refinement: mid-episode command resampling inside the range (seconds).
  refine_resample_time: tuple[float, float] = (4.0, 8.0)

  # Success test (SGS): |mean(v) - v_cmd| <= max(abs, rel * v_cmd), after settle_time.
  success_err_abs: float = 0.1
  success_err_rel: float = 0.1
  success_err_yaw: float = 0.2  # rad/s, |mean yaw rate| (straight line => 0 commanded)
  # Footfall timing (SGS success): mean over strides of the worst deviation, in cycle
  # fractions, of the left/right and hind->fore touchdown spacings from the clock's.
  # None = not part of the success test.
  success_timing_tol: float | None = None
  settle_time: float = 1.0

  # Terminations. Tune per gait when designing rewards (gallop pitches a lot and has
  # flight phases; walk-level limits would kill good gallop episodes).
  max_tilt_deg: float = 60.0
  min_base_height: float = 0.30

  tracking_std: float = 0.5  # m/s, lin-vel tracking kernel width (placeholder)
  budget: StageBudget = field(default_factory=StageBudget)

  # Control rate: physics substeps (SIM_DT each) per policy step. 4 = 50 Hz. The PPO
  # discount, GAE lambda and rollout length are rescaled from it (rl/cfg.py) so their
  # horizons in seconds stay the same at every rate.
  decimation: int = 4

  @property
  def control_dt(self) -> float:
    return self.decimation * SIM_DT

  @property
  def v_min(self) -> float:
    return self.speed_range[0]

  @property
  def v_max(self) -> float:
    return self.speed_range[1]

  @property
  def sgs_speed_range(self) -> tuple[float, float]:
    return (max(self.v_min, self.sgs_min_speed), self.v_max)

  def stride_frequency(self, v: float) -> float:
    return self.stride(v, self.clock.duty_factor)

  def duty_at(self, v: float) -> float:
    return self.clock.duty_at(self.stride_frequency(v))

  @property
  def cycle_time_range(self) -> tuple[float, float]:
    """(shortest, longest) cycle in seconds over the gait's speed range."""
    lo, hi = self.sgs_speed_range
    return (1.0 / self.stride_frequency(hi), 1.0 / self.stride_frequency(lo))

  @property
  def title(self) -> str:
    return self.name.capitalize()


##
# The four gaits. Speed ranges are the user's (m/s). Clocks / stride / terminations
# are PLACEHOLDERS to be revisited gait by gait with the rewards.
##

WALK = GaitSpec(
  name="walk",
  index=0,
  speed_range=(0.0, 1.30),  # Froude 0.26: horses walk up to ~0.3 (Robilliard et al. 2007)
  # Lateral sequence RR -> FR -> RL -> FL, footfalls 0.25 cycle apart (gait.json's order).
  # Air time per foot is fixed (horse swing time barely changes with speed, ~0.25 s
  # scaled to this leg) and the duty factor follows the stride frequency: 0.79 at
  # 0.15 m/s down to 0.58 at 1.3 m/s, so there are always 2-3 feet down.
  clock=GaitClock(
    touchdown=(0.476, 0.976, 0.226, 0.726),
    duty_factor=0.65,
    air_time=0.25,
    duty_min=0.55,
    duty_max=0.80,
  ),
  # f = 0.7 + 0.95 v, capped at 1.7 Hz: a planted foot sweeps duty * v / f = 0.14 m at
  # 0.15 m/s and 0.44 m at 1.3 m/s (the walking task's 0.15-0.45 m stance reach).
  stride=StrideFrequency(stance_sweep=0.35, f_min=0.6, f_max=1.7, slope=0.95, offset=0.7),
  speed_bins=10,
  sgs_min_speed=0.15,  # below this the command is a stand, not a walk
  spawn_from_rest=True,
  random_start_phase=False,
  max_tilt_deg=60.0,
  tracking_std=0.35,
  success_timing_tol=0.05,
)

TROT = GaitSpec(
  name="trot",
  index=1,
  speed_range=(1.20, 2.80),  # Froude 0.25: horses start trotting here
  # Diagonal pairs. As in the horse trot (Robilliard et al. 2007: swing time hardly
  # changes with speed, stance shrinks) each foot's air time is held at ~0.24 s (horse
  # swing ~0.345 s scaled by sqrt(0.58 / 1.2), see docs/multi_policy.md) and the duty
  # factor follows the stride frequency: 0.47 at 1.2 m/s -> 0.39 at 2.8 m/s. The flight
  # between the two stances is (0.5 - duty) / f: 16 ms at 1.2 m/s growing to 43 ms.
  clock=GaitClock(
    touchdown=(0.0, 0.5, 0.5, 0.0),
    duty_factor=0.45,
    air_time=0.24,
    duty_min=0.35,
    duty_max=0.50,
  ),
  # f = 2.0 + 0.19 v: 2.23 Hz at 1.2 m/s, 2.53 Hz at 2.8 m/s (horse-scaled trot, x1.44).
  stride=StrideFrequency(stance_sweep=0.40, f_min=2.0, f_max=3.0, slope=0.19, offset=2.0),
  speed_bins=14,
  max_tilt_deg=60.0,
  tracking_std=0.5,
  # Footfall timing (SGS success): horse trot ratios are 0.50 +- 0.03, so 2 s.d.
  success_timing_tol=0.06,
)

CANTER = GaitSpec(
  name="canter",
  index=2,
  speed_range=(2.24, 6.00),
  # Right-lead 3-beat: trailing hind (RL), diagonal pair (FL, then RR), leading fore (FR),
  # then one flight. Spacings from the horse canter (Robilliard et al. 2007, Table 4,
  # right canter): FR - FL = 0.27 +- 0.03 (0.28 here) and RR - RL = 0.33 (the paper's 0.67
  # is |RH - LH| measured from LF). FL lands just BEFORE RR: if the diagonal hind landed
  # first the footfalls would be a gallop's (Table 1), so the order is part of the gait.
  # As for the trot, each foot's air time is held (horse swing ~300 ms x 0.69 = 0.20 s)
  # and the duty factor follows the stride frequency. The single flight is 0.40 - duty
  # of the cycle after FR lifts: 8 ms at 2.24 m/s (duty capped at 0.38) -> 23 ms at 6 m/s.
  # duty_min = 0.32 = the FL touchdown: below it RL would lift before the diagonal pair
  # lands and the stride would get a second flight phase (not a canter). At most 3 feet
  # are ever down (Table 1: 0-3).
  clock=GaitClock(
    touchdown=(0.60, 0.32, 0.33, 0.0),
    duty_factor=0.36,
    air_time=0.20,
    duty_min=0.32,
    duty_max=0.38,
  ),
  # Horse f ~ 1.42 + 0.12 v over 5-9 m/s, x1.44 for the leg length: f = 2.04 + 0.25 v
  # (2.60 Hz at 2.24 m/s, 3.54 Hz at 6 m/s). 2.74 Hz at 2.8 m/s vs the trot's 2.53 Hz.
  stride=StrideFrequency(stance_sweep=0.45, f_min=2.4, f_max=3.8, slope=0.25, offset=2.04),
  speed_bins=15,
  max_tilt_deg=70.0,
  tracking_std=0.8,
  # Footfall timing (SGS success): horse canter ratios are 0.27 +- 0.03 (fore) and
  # +- 0.05 (hind), so ~2 s.d.
  success_timing_tol=0.08,
)

GALLOP = GaitSpec(
  name="gallop",
  index=3,
  # "4.47 onwards": the top end is OPEN in the spec. 8 m/s (Froude ~11) is a
  # placeholder ceiling for the ladder; raise it if the policy saturates.
  speed_range=(4.47, 8.00),
  # Right-lead transverse 4-beat: RL (non-lead hind), RR (lead hind), FL (non-lead fore),
  # FR (lead fore), then ONE flight. From Witte, Hirst & Wilson 2006 (4389.pdf, 9-17 m/s
  # Thoroughbreds) scaled by Froude similarity with a 1.5 m horse leg: speeds and
  # frequencies x sqrt(1.5 / 0.58) = 1.61, times x 0.62, so 4.47-8 m/s here is a horse
  # galloping at ~7.2-12.9 m/s. See docs/multi_policy.md (Gallop) for the derivation.
  #   - Swing time is nearly constant (fore 364 -> 342 ms, hind 355 -> 326 ms), hind
  #     stance is always longer than fore: per-leg air time fore 0.225 s, hind 0.217 s, so
  #     duty fore 0.30 -> 0.20, hind 0.33 -> 0.23 (horse fits 0.28 -> 0.21, 0.32 -> 0.24).
  #   - Every consecutive pair of footfalls overlaps (no double suspension in 5642
  #     strides) and the single flight is ~27% of the stride. Touchdowns chosen so the
  #     three footfall overlaps stay > 0 over the whole range (50 ms each at 4.47 m/s ->
  #     ~15 ms at 8 m/s; horse total 183 -> 35 ms), one flight of 64 ms at 4.47 m/s ->
  #     84 ms at 8 m/s (horse 135 -> 119 ms x 0.62 = 84 -> 74 ms), at most 2 feet down.
  #   (The previous placeholder, touchdowns .57/.45/.12/0 with duty 0.28, had a second
  #   flight between RR lift-off at 0.40 and FL touchdown at 0.45.)
  clock=GaitClock(
    touchdown=(0.50, 0.345, 0.17, 0.0),
    duty_factor=0.26,
    air_time=(0.225, 0.225, 0.217, 0.217),
    duty_min=0.19,
    duty_max=0.33,
  ),
  # Horse stride frequency rises only 2.02 -> 2.41 Hz over 9-17 m/s (speed comes mostly
  # from stride length): f_h ~ 1.58 + 0.049 v_h, scaled: f = 2.54 + 0.127 v = 3.11 Hz at
  # 4.47 m/s (canter 3.16 Hz there: continuous handoff) -> 3.56 Hz at 8 m/s.
  stride=StrideFrequency(stance_sweep=0.50, f_min=3.0, f_max=3.8, slope=0.127, offset=2.54),
  speed_bins=14,
  max_tilt_deg=75.0,
  min_base_height=0.25,
  tracking_std=1.0,
  # Stance is 60-100 ms: 3-5 steps at 50 Hz, and the top-speed footfall overlap is ~16 ms.
  # 100 Hz control (decimation 2) for now.
  decimation=2,
  # One 100 Hz step is 0.031-0.036 of a cycle; ~3 steps of slack on the worst spacing.
  success_timing_tol=0.10,
)

# Training order. Each gait forks from the previous one's ladder.
GAITS: tuple[GaitSpec, ...] = (WALK, TROT, CANTER, GALLOP)
GAIT_BY_NAME: dict[str, GaitSpec] = {g.name: g for g in GAITS}
NUM_GAITS = 4  # one-hot width; fixed so the obs layout never changes when gaits are added


def neighbours(spec: GaitSpec) -> tuple[GaitSpec | None, GaitSpec | None]:
  """(slower gait, faster gait) in the training order."""
  i = GAITS.index(spec)
  return (GAITS[i - 1] if i > 0 else None, GAITS[i + 1] if i + 1 < len(GAITS) else None)


def check_gait_chain(gaits: tuple[GaitSpec, ...] = GAITS) -> list[str]:
  """Warnings about the speed envelopes (gaps break the handoff between policies)."""
  warnings = []
  for slow, fast in zip(gaits, gaits[1:]):
    if fast.v_min > slow.v_max:
      warnings.append(
        f"{slow.name} -> {fast.name}: GAP {slow.v_max:.2f}..{fast.v_min:.2f} m/s. No policy "
        f"covers it, so an upshift hands a {slow.v_max:.2f} m/s state to a gait commanded "
        f">= {fast.v_min:.2f} m/s (it must accelerate across the gap) and a downshift "
        f"must brake across it."
      )
    else:
      warnings.append(
        f"{slow.name} -> {fast.name}: overlap {fast.v_min:.2f}..{slow.v_max:.2f} m/s "
        f"(hysteresis band for switching)."
      )
  assert sorted(g.index for g in gaits) == list(range(len(gaits)))
  assert max(g.index for g in gaits) < NUM_GAITS
  return warnings


def describe() -> str:
  lines = []
  for g in GAITS:
    lo, hi = g.sgs_speed_range
    lines.append(
      f"{g.name:7s} v={g.v_min:.2f}-{g.v_max:.2f} m/s (Fr {froude(g.v_min):.2f}-"
      f"{froude(g.v_max):.2f})  duty={g.duty_at(lo):.2f}-{g.duty_at(hi):.2f}  "
      f"f={g.stride_frequency(lo):.2f}-{g.stride_frequency(hi):.2f} Hz  "
      f"bins={g.speed_bins}  touchdown={g.clock.touchdown}  "
      f"control={1.0 / g.control_dt:.0f} Hz"
    )
  return "\n".join(lines + check_gait_chain())


if __name__ == "__main__":
  print(describe())
