"""Gait command: straight-line forward speed + the gait's foot-contact clock.

The command is (lin_x, 0, 0): no lateral and no yaw command, ever. On top of the
velocity the term runs the gait clock (single source of truth for every reward,
observation and metric that needs the schedule):

  cycles  gait cycles since the episode started, integrated from the stride frequency
          f(v_cmd) (gaits.StrideFrequency), so a speed change changes the clock rate,
          never its phase.
  stance  leg i is scheduled on the ground while ((cycles + offset_i) mod 1) < duty_i
          (duty is per leg: equal for every leg unless the gait's air_time is per leg).

Speeds per episode come from SGS when it is active (env.sgs), otherwise uniformly from
the gait's range (play, handoff recording). The reset event (spawning the robot already
moving) needs the episode's speed BEFORE the command manager resets, so the first speed
of each episode is drawn by `draw_episode_speed`, called from the reset event, and the
command reset then uses it.

Mid-episode resampling (refinement) draws a new speed uniformly over the range, up or
down, so the policy practises speed changes inside its gait.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class GaitCommand(CommandTerm):
  cfg: GaitCommandCfg

  def __init__(self, cfg: GaitCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    n, dev = self.num_envs, self.device

    # Clock tables in contact-sensor (model) foot order.
    _, sensor_order = self.robot.find_geoms(list(cfg.foot_geoms))
    perm = [list(cfg.foot_geoms).index(name) for name in sensor_order]
    self.foot_order = tuple(sensor_order)
    self.offsets = torch.tensor(cfg.offsets, device=dev)[perm].view(1, -1)
    self.num_feet = len(perm)
    # Set per episode by the reset event: True when the robot spawned at rest (not from a
    # handoff bank and not already moving).
    self.from_rest = torch.zeros(n, dtype=torch.bool, device=dev)

    self.vel_command_b = torch.zeros(n, 3, device=dev)
    self.stride_freq = torch.full((n,), cfg.f_min, device=dev)
    # [B, feet] per-leg duty (sensor order), follows the speed when air_time is set.
    self.duty = torch.full((n, self.num_feet), cfg.duty_factor, device=dev)
    self._air_time = None
    if cfg.air_time is not None:
      air = cfg.air_time if isinstance(cfg.air_time, tuple) else (cfg.air_time,) * self.num_feet
      self._air_time = torch.tensor(air, device=dev)[perm].view(1, -1)
    self.gait_cycles = torch.zeros(n, device=dev)
    self.gait_onehot = torch.zeros(n, cfg.num_gaits, device=dev)
    self.gait_onehot[:, cfg.gait_index] = 1.0

    self._next_speed = torch.zeros(n, device=dev)
    self._has_next = torch.zeros(n, dtype=torch.bool, device=dev)

    # Episode accumulators for the SGS success test (signed errors, after settle time).
    self.ep_err_x = torch.zeros(n, device=dev)
    self.ep_yaw = torch.zeros(n, device=dev)
    self.ep_cmd = torch.zeros(n, device=dev)
    self.ep_steps = torch.zeros(n, device=dev)
    self.ep_time = torch.zeros(n, device=dev)

    # Footfall timing (only when cfg.track_footfalls): last touchdown time of each foot,
    # the previous front-left touchdown (stride start) and the per-episode error sums.
    self.last_td = torch.full((n, self.num_feet), -1.0, device=dev)
    self.prev_fl = torch.full((n,), -1.0, device=dev)
    self.ep_tim_err = torch.zeros(n, device=dev)
    self.ep_strides = torch.zeros(n, device=dev)
    if cfg.track_footfalls:
      idx = {leg: next(i for i, g in enumerate(self.foot_order) if g.startswith(leg)) for leg in ("FR", "FL", "RR", "RL")}
      self._leg = idx
      td = (-self.offsets[0]) % 1.0  # touchdown cycle fraction, sensor order
      # Expected spacings (cycle fractions): FR after FL, RR after RL, FL after RL, FR after RR.
      self._expected = torch.stack(
        [(td[idx["FR"]] - td[idx["FL"]]) % 1.0, (td[idx["RR"]] - td[idx["RL"]]) % 1.0,
         (td[idx["FL"]] - td[idx["RL"]]) % 1.0, (td[idx["FR"]] - td[idx["RR"]]) % 1.0]
      )

    self.metrics["error_vel_x"] = torch.zeros(n, device=dev)
    self.metrics["error_vel_yaw"] = torch.zeros(n, device=dev)
    self.metrics["error_vel_y"] = torch.zeros(n, device=dev)

  # ------------------------------------------------------------------ public API
  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  @property
  def speed(self) -> torch.Tensor:
    return self.vel_command_b[:, 0]

  def is_moving(self) -> torch.Tensor:
    return self.speed > self.cfg.move_threshold

  def frequency_of(self, v: torch.Tensor) -> torch.Tensor:
    c = self.cfg
    f = c.f_offset + c.f_slope * v if c.f_slope is not None else v * c.duty_factor / c.stance_sweep
    return f.clamp(c.f_min, c.f_max)

  def duty_of(self, f: torch.Tensor) -> torch.Tensor:
    """[B] stride frequencies -> [B, feet] per-leg duty factors."""
    c = self.cfg
    if self._air_time is None:
      return torch.full((f.shape[0], self.num_feet), c.duty_factor, device=f.device)
    return (1.0 - self._air_time * f.unsqueeze(1)).clamp(c.duty_min, c.duty_max)

  def mean_duty(self) -> torch.Tensor:
    """[B] duty averaged over the legs (the 1-wide gait_duty observation)."""
    return self.duty.mean(dim=1)

  def scheduled_flight_fraction(self, samples: int = 100) -> torch.Tensor:
    """[B] share of the cycle in which the clock schedules no foot in stance."""
    grid = (torch.arange(samples, device=self.device) + 0.5) / samples  # [G]
    phase = (grid.view(1, -1, 1) + self.offsets.unsqueeze(1)) % 1.0  # [1, G, feet]
    stance = phase < self.duty.unsqueeze(1)  # [B, G, feet]
    return (~stance.any(dim=2)).float().mean(dim=1)

  def cycles_at(self, lookahead_steps: int = 0) -> torch.Tensor:
    """[B] gait cycles, advanced by lookahead_steps control steps (only while moving)."""
    cycles = self.gait_cycles
    if lookahead_steps:
      dt = lookahead_steps * self._env.step_dt * self.stride_freq
      cycles = cycles + torch.where(self.is_moving(), dt, 0.0)
    return cycles

  def leg_phase(self, lookahead_steps: int = 0) -> torch.Tensor:
    """[B, feet] cycle fraction of each leg (0 = touchdown). Lookahead in control steps."""
    return (self.cycles_at(lookahead_steps).unsqueeze(1) + self.offsets) % 1.0

  def stance_target(self, lookahead_steps: int = 1) -> torch.Tensor:
    """[B, feet] bool scheduled stance. Default lookahead 1: rewards are computed
    before the clock ticks, and the action being rewarded was chosen one step ago.

    In an episode started from rest every foot is held down until its first lift-off
    (startup hold), so the gait begins one leg at a time instead of with two feet
    snatched off the ground at t = 0."""
    duty = self.duty
    stance = self.leg_phase(lookahead_steps) < duty
    # First lift-off of each leg, in cycles: when its phase reaches the duty factor.
    first = (duty - self.offsets) % 1.0
    first = torch.where(first > 1.0 - 1e-6, torch.zeros_like(first), first)
    cycles = self.cycles_at(lookahead_steps).unsqueeze(1)
    held = self.from_rest.unsqueeze(1) & (cycles < first)
    return stance | held

  def clock_obs(self) -> torch.Tensor:
    """[B, 2 * feet]: sin/cos of each leg's phase (zero when not moving)."""
    angle = 2.0 * math.pi * self.leg_phase()
    obs = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)
    return obs * self.is_moving().float().unsqueeze(1)

  def draw_episode_speed(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Draw (once) the first commanded speed of the episode starting in env_ids."""
    sgs = getattr(self._env, "sgs", None)
    if sgs is not None and sgs.command_name == self.cfg.command_name:
      v = sgs.speed_of(env_ids)
    else:
      v = self._uniform_speed(len(env_ids))
    self._next_speed[env_ids] = v
    self._has_next[env_ids] = True
    return v

  def episode_outcome(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """(mean signed vx error, mean yaw rate, mean |v_cmd|, counted seconds)."""
    n = self.ep_steps[env_ids].clamp(min=1)
    return (
      self.ep_err_x[env_ids] / n,
      self.ep_yaw[env_ids] / n,
      self.ep_cmd[env_ids] / n,
      self.ep_steps[env_ids] * self._env.step_dt,
    )

  # ------------------------------------------------------------------ internals
  def _uniform_speed(self, k: int) -> torch.Tensor:
    lo, hi = self.cfg.speed_range
    return lo + (hi - lo) * torch.rand(k, device=self.device)

  def _set_speed(self, env_ids: torch.Tensor, v: torch.Tensor) -> None:
    self.vel_command_b[env_ids] = 0.0
    self.vel_command_b[env_ids, 0] = v
    f = self.frequency_of(v)
    self.stride_freq[env_ids] = f
    self.duty[env_ids] = self.duty_of(f)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    starting = self.command_counter[env_ids] == 0  # episode reset (see CommandTerm.reset)
    if starting.any():
      ids = env_ids[starting]
      missing = ~self._has_next[ids]
      if missing.any():
        self.draw_episode_speed(ids[missing])
      self._set_speed(ids, self._next_speed[ids])
      self._has_next[ids] = False
      if self.cfg.random_start_phase:
        self.gait_cycles[ids] = torch.rand(len(ids), device=self.device)
      else:
        self.gait_cycles[ids] = 0.0
      for buf in (self.ep_err_x, self.ep_yaw, self.ep_cmd, self.ep_steps, self.ep_time,
                  self.ep_tim_err, self.ep_strides):
        buf[ids] = 0.0
      self.last_td[ids] = -1.0
      self.prev_fl[ids] = -1.0
    if (~starting).any():
      ids = env_ids[~starting]
      if self.cfg.resample_mid_episode:
        self._set_speed(ids, self._uniform_speed(len(ids)))

  def _update_command(self) -> None:
    dt = self._env.step_dt
    self.gait_cycles = torch.where(
      self.is_moving(), self.gait_cycles + dt * self.stride_freq, self.gait_cycles
    )

  def _update_metrics(self) -> None:
    dt = self._env.step_dt
    max_steps = self.cfg.resampling_time_range[1] / dt
    v = self.robot.data.root_link_lin_vel_b
    w = self.robot.data.root_link_ang_vel_b
    err_x = v[:, 0] - self.vel_command_b[:, 0]
    self.metrics["error_vel_x"] += err_x.abs() / max_steps
    self.metrics["error_vel_y"] += v[:, 1].abs() / max_steps
    self.metrics["error_vel_yaw"] += w[:, 2].abs() / max_steps

    self.ep_time += dt
    counted = (self.ep_time > self.cfg.settle_time).float()
    self.ep_err_x += counted * err_x
    self.ep_yaw += counted * w[:, 2]
    self.ep_cmd += counted * self.vel_command_b[:, 0].abs()
    self.ep_steps += counted
    if self.cfg.track_footfalls:
      self._update_footfalls(counted > 0)

  def _update_footfalls(self, counted: torch.Tensor) -> None:
    """One stride = front-left touchdown to the next. At each one, compare the spacing of
    the other feet's last touchdowns with the clock's (Robilliard et al. 2007 use exactly
    these: fore and hind left/right ratios 0.5, hind->fore advance 0.25 for the walk).
    A foot that did not land during the stride counts as the worst error (0.5)."""
    sensor = self._env.scene[self.cfg.contact_sensor]
    first = sensor.compute_first_contact(dt=self._env.step_dt)  # [B, feet]
    t = self.ep_time
    L = self._leg
    self.last_td = torch.where(first, t.unsqueeze(1), self.last_td)
    stride = first[:, L["FL"]] & (self.prev_fl >= 0)
    if stride.any():
      T = (t - self.prev_fl).clamp(min=1e-3)
      td = self.last_td
      ratios = torch.stack(
        [((td[:, L["FR"]] - t) / T) % 1.0, ((td[:, L["RR"]] - td[:, L["RL"]]) / T) % 1.0,
         ((t - td[:, L["RL"]]) / T) % 1.0, ((td[:, L["FR"]] - td[:, L["RR"]]) / T) % 1.0], dim=1)
      dev = ((ratios - self._expected + 0.5) % 1.0 - 0.5).abs().max(dim=1).values
      landed = (td[:, [L["FR"], L["RR"], L["RL"]]] > self.prev_fl.unsqueeze(1)).all(dim=1)
      dev = torch.where(landed, dev, torch.full_like(dev, 0.5))
      take = stride & counted
      self.ep_tim_err += torch.where(take, dev, 0.0)
      self.ep_strides += take.float()
    self.prev_fl = torch.where(first[:, L["FL"]], t, self.prev_fl)

  def timing_outcome(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean worst footfall-spacing deviation per stride, strides counted)."""
    n = self.ep_strides[env_ids]
    return self.ep_tim_err[env_ids] / n.clamp(min=1), n


@dataclass(kw_only=True)
class GaitCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  command_name: str = "gait"  # this term's key in cfg.commands (SGS looks it up)
  gait_name: str
  gait_index: int
  num_gaits: int
  speed_range: tuple[float, float]
  # Clock (LEG order, matching foot_geoms).
  foot_geoms: tuple[str, ...]
  offsets: tuple[float, ...]
  duty_factor: float
  # Duty factor follows the stride frequency when air_time is set (see GaitClock); a
  # tuple is one air time per leg (LEG order, matching foot_geoms).
  air_time: float | tuple[float, ...] | None = None
  duty_min: float = 0.0
  duty_max: float = 1.0
  # Stride frequency f(v) = clamp(v * duty / stance_sweep, f_min, f_max), or
  # clamp(f_offset + f_slope * v, f_min, f_max) when f_slope is set.
  f_slope: float | None = None
  f_offset: float = 0.0
  track_footfalls: bool = False
  contact_sensor: str = "feet_ground_contact"
  stance_sweep: float
  f_min: float
  f_max: float
  random_start_phase: bool = True
  resample_mid_episode: bool = False
  move_threshold: float = 0.05
  settle_time: float = 1.0

  def build(self, env: ManagerBasedRlEnv) -> GaitCommand:
    return GaitCommand(self, env)
