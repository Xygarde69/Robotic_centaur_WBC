# Multi-gait pipeline (`centaur_rl/tasks/Multi_Policy`)

One command trains four gait policies in sequence: walk → trot → canter → gallop. The
command is forward speed on flat ground, never `lin_y`. Handoff is straight. The refine stage adds
non-holonomic turns: `ang_z = lin_x_cmd / r` (see [Turns](#turns-refine-only)).
Each gait gets its own policy. Every gait after the walk starts from the previous gait's
weights.

```bash
cd scripts
python train.py --multi-Gait_straight_flat_SGS --env.scene.num-envs 4096          # everything
python train.py --task Multi-Gait_straight_flat_SGS --dry-run                     # print the plan
python train.py --task Multi-Gait_straight_flat_SGS --run-name R --resume         # continue run R
python train.py --task Multi-Gait_straight_flat_SGS --gaits walk trot --handoff-iterations 0
python train.py --task Multi-Gait_straight_flat_SGS --init-checkpoint <walk refine model_N.pt>  # walk refine starts from it
python train.py --task Multi-Gait_straight_flat_SGS-Trot-Refine --agent.stage.init-mode fork \
    --agent.stage.init-checkpoint <walk refine model_N.pt>                         # one stage by hand
python play.py Multi-Gait_straight_flat_SGS-Trot-Refine --checkpoint-file <model.pt>
python play.py Multi-Gait_straight_flat_SGS-Walk-Refine --checkpoint-file <model.pt> --speed 0.5
```

`play.py --radius R` holds a turn of R m (+ left, − right, 0 straight); refine checkpoints otherwise get a random
feasible radius per episode. `play.py --joystick` (Refine tasks) steers with a gamepad over the Linux joystick API
(`tools/joystick.py`, no extra packages): hold L1 to turn left, R1 to turn right; the stick ramps to full lock in 1.5 s and
returns to 0 in 0.5 s after release. Curvature = stick * min(1/r_min, 0.4 g / v^2), ang_z = v * curvature, so it stays
within what the curriculum trains. `play.py --speed V` holds the command at V m/s (no resampling). A **ladder** checkpoint plays at a constant speed by
default: the fastest speed it had mastered when saved, i.e. the lower edge of its SGS frontier bin (the checkpoint
stores the SGS state). With no bin mastered that is the bottom of the gait's range. `--speed` overrides it. Other
checkpoints play with the speed resampled over the gait's range every 4–8 s.

`--stage-init JOB=PATH` restarts one job from a checkpoint in continue mode (actor, critic, optimizer; no warmup or
std floor), e.g. `--stage-init trot-refine=<trot model_N.pt>` after a clock or reward change. Set that job back to
pending in `pipeline.json` (or delete its entry) before `--resume` if it was marked done.

Unknown options are forwarded to every training stage (`--env.*`, `--agent.*`, `--gpu-ids`, ...).

## Play all four gaits: `scripts/Multiplay.py`

```bash
cd scripts && python Multiplay.py              # bundled policies in policies/multigait/
python Multiplay.py --run gaits_v4             # or a pipeline run's policies
python Multiplay.py --gallop <model_N.pt>      # override one gait
```

DualSense (Linux joystick API, `centaur_rl/tasks/Multi_Policy/tools/dualsense.py`, buttons and axes looked up
by kernel code): **R2** hold = speed up (rate ∝ trigger), **L2** = slow down, **Triangle** = next gait up,
**Circle** = next gait down, **L1 / R1** = turn left / right. One simulation at 100 Hz (the gallop's rate); the
50 Hz walk / trot / canter policies are queried every second step with the action held, as in training. A gait
switch swaps the gait command's clock live (the stride phase carries on) and clamps the speed into the new gait's
range: shift inside the overlaps (walk/trot 1.20–1.30, trot/canter 2.24–2.80, canter/gallop 4.47–6.00 m/s) for
a smooth change. A fall respawns the robot at rest in the walk. `policies/multigait/manifest.json` lists where
each bundled checkpoint came from.

## Stages per gait

| Stage | What it trains | Starts from |
|---|---|---|
| `g-refine` | whole speed range × turn radius (2 km → R_min, per-speed radius frontier), speed resampled every 4–8 s, up and down, radius kept. 20% of resets start from the slower gait's top-speed states. | walk: from scratch (or `--init-checkpoint`, continue). Other gaits: **fork** of the previous gait's refine checkpoint (`fork_iteration`, default its last one) |
| `g-up` / `g-down` | record handoff banks: states of `g-refine` at the gait's top or lowest speed | — |
| `g-handoff` (optional, 500 it) | as refine, with 50% of resets from neighbour banks (slower gait's `up`, faster gait's `down`) | continue from `g-refine` |

**Fork** (new gait):
- The actor weights and the actor's observation normalizer are loaded; the critic and optimizer start fresh.
- The actor is frozen for the first 50 iterations (was 200; the trot critic's value loss settles in ~40–50), so only
  the critic trains. The learning rate is held at the base 1e-3 during the warmup (schedule `fixed`), and the adaptive
  schedule resumes from 1e-3 at the first unfrozen update. Before this fix the schedule climbed to its 1e-2 cap while
  the actor was frozen (KL ≈ 0). The first actor update then ran at 10× the base rate and wrecked the gaits_v4 trot
  fork in one iteration (episode length 148 → 5 steps).
- Action std is raised to at least 0.4 (was 0.6: in gaits_v4 the extra noise inflated the trot fork's action-rate
  and slip penalties).
- Trot, canter and gallop get an **`is_alive`** bonus (+0.75 per second, `rewards/common.py`). The first gaits_v4 trot
  fork netted about −4.1 per second alive, against a one-off −4 for terminating (`is_terminated` −200 × 0.02 s). So it
  learned to drop the base within ~0.2 s (all `base_too_low`, std 0.6 → 0.21 by it 350). The walk has no bonus: it
  never forks. Tried 5, 1 and 0.1, now 0.75. With +1, the second fork learned to **stand**
  (v_x ≈ −0.05 m/s): standing scored ~+3.1 per second (`track_angular_velocity` 0.97, `is_alive` 1.0, `pose` 0.79,
  `track_turn_lean` 0.49, `track_yaw_fine` 0.45), while the linear kernel gives no gradient 2 m/s from the command.
- **Progress gate (every gait):** `track_angular_velocity`, `track_yaw_fine` and `track_turn_lean` are multiplied by
  `forward_progress` = clamp(v_x / v_cmd, 0, 1) (`gate_by_progress=True`). Standing still earns none of them, and
  together they grow linearly with the speed reached, which is the gradient out of a standstill.
- The actor normalizer's sample count is capped at 2M, so its mean and std re-adapt to the new gait. It is never frozen.

**Continue** (same gait): actor, critic and optimizer are all loaded.

Default budgets are in `gaits.StageBudget`: refine 10000 (walk 5000), handoff 500, critic warmup 50.

There is no ladder stage in the pipeline. In run `gaits_v2` the walk ladder stayed at 0% success in its first
speed bin for all 2000 iterations, and its speed error grew to 0.63 m/s by the end. The trot forked from it never tracked speed (1.2–1.9 m/s error). The
walk refine learned to walk (0.07 m/s error). The `-Ladder` stage tasks are still registered for training by hand.

Each gait's deployed policy is the final checkpoint of `g-handoff`, or of `g-refine` if
handoff is disabled. The paths are written to `pipeline.json` → `deployed_policies`.

## Gaits (`gaits.py`)

| Gait | Speed (m/s) | Clock (touchdown FR, FL, RR, RL / duty) | Spawn |
|---|---|---|---|
| walk | 0–1.30 (SGS from 0.15) | .476, .976, .226, .726 (lateral sequence, footfalls 0.25 apart) / 0.79 → 0.57 with speed | at rest |
| trot | 1.20–2.80 | 0, .5, .5, 0 (diagonal pairs) / 0.38 → 0.29 with speed | moving at the command |
| canter | 2.24–6.00 | 0.51, .272, .281, 0 (right lead: RL, FL → RR, FR) / 0.34 → 0.272 with speed | moving |
| gallop | 4.47–8.00 (top end is a placeholder) | .50, .345, .17, 0 (right lead: RL, RR, FL, FR, one flight) / fore 0.30 → 0.20, hind 0.33 → 0.23 | moving |

Walk/trot, trot/canter and canter/gallop all overlap (`check_gait_chain`). The walk tops out at Froude 0.30
and the trot starts at 0.25, where horses change gait (Robilliard et al. 2007, 187.pdf).

Stride frequency rises with speed. The walk uses `f(v) = clamp(0.7 + 0.95 v, 0.6, 1.7)` Hz (strides lengthen faster than they quicken, as in
horses), and its duty factor follows the frequency with a fixed air time per foot:
`duty = clamp(1 − 0.25 s · f, 0.55, 0.80)` (0.79 at 0.15 m/s, 0.57 at 1.3 m/s, always 2–3 feet down). Set per gait
by `GaitClock.air_time` and `StrideFrequency.slope/offset`; `gait_duty` is the per-env duty, so the observation
layout is unchanged. The trot does the same with `f = 2.0 + 0.19 v` (2.0–3.0 Hz) and `duty = clamp(1 − 0.28 s · f,
0.28, 0.50)` (0.38 at 1.2 m/s, 0.29 at 2.8 m/s). The canter uses `f = 2.04 + 0.25 v` (2.60–3.54 Hz) and
`duty = clamp(1 − 0.22 s · f, 0.272, 0.34)`. The gallop uses `f = 2.54 + 0.127 v` (3.11–3.56 Hz) and a
per-leg air time (fore 0.225 s, hind 0.217 s), so fore and hind duty differ; `gait_duty` shows their mean.

Fixed for every gait, because changing any of these invalidates every checkpoint:
- the observation layout
- action scale 0.5
- kp 150 / kd 5
- torque limit 400 Nm (sim only)

Control rate is per gait (`GaitSpec.decimation`, physics 0.005 s): 50 Hz for walk, trot and canter, **100 Hz for
the gallop** (its 60–100 ms stances are only 3–5 steps at 50 Hz). The PPO discount, GAE λ and rollout length are
rescaled so their horizons stay the same in seconds (`rl/cfg.py`): γ 0.99 → 0.995, λ 0.95 → 0.975, 24 → 48 steps
per env at 100 Hz. The gallop forks from the 50 Hz canter: the observation layout is identical, but the policy
now acts every 10 ms, which the critic warmup and std floor have to absorb.

## Observations

Actor:
- base angular velocity
- projected gravity
- `gait_command` [v_cmd, f, ω_cmd] (ω_cmd = 0 except on refine turns)
- `gait_clock`: sin/cos of each leg's phase (8 values)
- `gait_onehot` (4)
- `gait_duty` (1)
- joint positions and velocities
- last action

The critic gets all of the above plus:
- base linear velocity
- `gait_stance` (scheduled stance per foot)
- foot height, air time, contact and contact forces

## Rewards (`rewards/`)

`rewards/common.py` holds the shared PLACEHOLDER terms:
- forward-velocity tracking
- `straight_line` (no vy, no yaw)
- generic clock match `gait_schedule`
- tilt, tilt rate and vertical velocity
- termination
- action rate, joint acceleration, joint limits, torque
- foot slip

Each gait's own design goes in `rewards/<gait>.py`.

### Walk (`rewards/walk.py`)

The walking task's term set and weights (see [walking.md](walking.md)). It replaces `common_rewards`.

| Term | Weight | Function | Notes |
|---|---|---|---|
| `track_linear_velocity` | +1.0 | `track_linear_velocity` | std 0.35 (`WALK.tracking_std`; the walking task used 0.5) |
| `track_angular_velocity` | +1.0 | `track_angular_velocity` | std 0.5, holds w_z = 0 |
| `track_yaw_fine` | +0.5 | `track_angular_velocity` | std 0.25, `xy_weight` 0 (shared, `rewards/common.py`) |
| `track_turn_lean` | +0.5 | `track_turn_lean` | exp(−(g_y − sin θ)²/0.07²), θ = atan(v²κ/g); shared, `rewards/common.py` |
| `foot_gait` | +1.5 | `gait_contact_schedule` | 0.5 · per-foot match + 0.5 · [all four match] |
| `foot_contact_in_swing` | −0.5 | `feet_contact_in_swing` | |
| `touchdown_timing` | +3.0 | `touchdown_timing` | at each touchdown exp(−d²/0.08²), d = cycle-fraction distance from the scheduled touchdown; sparse (~0.1 of the weight per step at full marks) |
| `body_orientation_l2` | −1.0 | `body_orientation_roll_pitch` | upright target (no turns) |
| `body_ang_vel` | −0.05 | `body_angular_velocity_roll_pitch` | |
| `angular_momentum` | −0.025 | `angular_momentum_penalty` | |
| `pose` | +1.0 | `variable_posture` | walking task's per-joint stds |
| `foot_clearance` | −0.3 | `feet_clearance` | target 0.13 m |
| `foot_slip` | −0.25 | `feet_slip` | |
| `soft_landing` | −1e-3 | `soft_landing` | |
| `stand_still` | −1.0 | `stand_still` | inactive: the walk is never commanded to stand |
| `torque_l2` | −1e-5 | `torque_l2` | not in the walking task; torque limit is 400 Nm here |
| `is_terminated`, `joint_acc_l2`, `joint_pos_limits` | −200, −2.5e-7, −10 | mjlab | |
| `action_rate_l2` | −0.2 | mjlab | 4× the walking task's −0.05: action scale 0.5 instead of 0.25 |

Startup hold: in an episode that spawns at rest (not from a handoff bank) each foot is scheduled down until its
first lift-off (FR at 0, RL 0.25, FL 0.5, RR 0.75 cycles), as in the walking task. This is in
`GaitCommand.stance_target`, so rewards, the critic's `gait_stance` and the metrics all see it. With the
speed-dependent duty the first lift-offs are computed from the live duty, still in the order FR, RL, FL, RR.

Walk success (SGS): besides speed tracking and yaw, an episode must keep the clock's footfall spacing. At each
front-left touchdown the command measures the spacing FL→FR, RL→RR (both 0.5) and RL→FL, RR→FR (both 0.25) as
fractions of the stride (the ratios Robilliard et al. report for the horse walk: 0.50 ± 0.01 fore, 0.50 ± 0.02
hind). A stride's score is its worst deviation (0.5 if a foot did not land in the stride); the episode needs at
least one stride and a mean score ≤ `WALK.success_timing_tol` (0.05). Logged as `footfall_timing_error`.
**Currently off:** `env_cfg._sgs` passes `timing_tol=None` for every gait (see [Turns](#turns-refine-only)); the
metric is still logged.

### Trot (`rewards/trot.py`)

The trot task's term set and weights (see [trot.md](trot.md)). It replaces `common_rewards`.

| Term | Weight | Function | Notes |
|---|---|---|---|
| `track_linear_velocity` | +1.0 | `track_linear_velocity` | kernel width max(0.5, 0.2·v_cmd) m/s (0.5 = `TROT.tracking_std` = the trot task's √0.25; widens above 2.5 m/s), z weight 0.5 (flight needs base bounce) |
| `track_angular_velocity` | +1.0 | `track_angular_velocity` | std √0.5, holds w_z = 0 |
| `track_yaw_fine` | +0.5 | `track_angular_velocity` | std 0.25, `xy_weight` 0 (shared, `rewards/common.py`) |
| `track_turn_lean` | +0.5 | `track_turn_lean` | exp(−(g_y − sin θ)²/0.07²), θ = atan(v²κ/g); shared, `rewards/common.py` |
| `foot_gait` | +0.5 | `gait_contact_schedule` | exact_weight 0: fraction of feet matching the clock (the trot task's `feet_gait`) |
| `touchdown_timing` | +1.5 | `touchdown_timing` | **new.** std 0.045 cycle: diagonal partners must land together (horse ratios 0.50 ± 0.03); sparse, 4 landings per cycle |
| `excess_contact` | −0.3 | `feet_excess_contact` | **new.** feet on the ground beyond 2 (a trot has 0–2) |
| `foot_suspension` | **+2.0** (trot task 0.3) | `feet_suspension` | 1 while the clock schedules a flight and no foot is down; that is 1 − 2·duty of the cycle, 0.25 at 1.2 m/s to 0.42 at 2.8 m/s. At 0.3 it was worth ≤ 0.06 per second and gaits_v4 skipped flight below ~2 m/s |
| `body_orientation_l2` | −1.0 | `body_orientation_roll_pitch` | upright target (no turns) |
| `body_ang_vel` | −0.05 | `body_angular_velocity_roll_pitch` | body-frame roll/pitch rate (trot task: world-frame x/y; equal near upright) |
| `angular_momentum` | −0.025 | `angular_momentum_penalty` | |
| `pose` | +1.0 | `variable_posture` | trot task's stds; running threshold = `TROT.v_min` so every trot command uses "running" |
| `foot_clearance` | −1.5 | `feet_clearance` | target 0.15 m, **divided by the commanded speed** (`speed_normalized`); −1.5 equals the trot task's −1.0 at 1.5 m/s |
| `foot_slip` | −0.25 | `feet_slip` | |
| `soft_landing` | −1e-3 | `soft_landing` | |
| `stand_still` | −1.0 | `stand_still` | inactive: the trot is never commanded to stand |
| `torque_l2` | −1e-5 | `torque_l2` | not in the trot task; kept from the walk (torque limit is 400 Nm here) |
| `is_terminated`, `joint_acc_l2`, `joint_pos_limits` | −200, −2.5e-7, −10 | mjlab | |
| `is_alive` | +0.75 | mjlab | per second alive; forked gaits only (see Fork) |
| `action_rate_l2` | −0.2 | mjlab | 4× the trot task's −0.05: action scale 0.5 instead of 0.25 |

Differences from the trot task's own clock (fixed 0.6 s period, flight fraction 0.02→0.14 with speed): here
`foot_gait`, `foot_suspension`, `touchdown_timing` and `excess_contact` read `GaitCommand.stance_target`, so the
flight phases follow `TROT.clock`. It holds each foot's air time at 0.28 s and lets the duty factor
follow the stride frequency, so flight (1 − 2·duty)/(2f) grows with speed: 56 ms at 1.2 m/s, 67 ms at 1.8,
83 ms at 2.8.

#### Horse-data additions

Reference: Robilliard, Pfau & Wilson 2007, *Gait characterisation and classification in horses*,
J. Exp. Biol. 210:187–197 (`187.pdf` in the repo root). Horse trot data (Table 2, Fig. 1, Table 4), 2–6.5 m/s:
- stride time 649 → 482 ms (1.54 → 2.07 Hz)
- LF stance 302 → 158 ms, while swing stays at ~310–350 ms
- duty factor ~0.47 → 0.33
- fore and hind left/right ratios 0.50 ± 0.03 and 0.49 ± 0.03
- 0–2 limbs on the ground, aerial phase "possible"
- hind stance only 1.6% shorter than fore

Scaling assumption: by Froude similarity (Fr = v²/gL), with L ≈ 1.2 m for an Icelandic horse versus 0.58 m here.
This L is an estimate, not a value from the paper. It matches the walk's 0.25 s air time, which is ~0.36 s of
horse swing scaled the same way. On that basis the robot's 1.2–2.8 m/s (Fr 0.25–1.38) corresponds to horse
trots at ~1.7–4.0 m/s. Times scale by √(0.58/1.2) ≈ 0.69 and frequencies by ≈ 1.44.

1. **Duty falls with speed at a fixed air time** (`gaits.TROT`): `GaitClock(air_time=0.28, duty_min=0.28,
   duty_max=0.50)` and `StrideFrequency(slope=0.19, offset=2.0, f_min=2.0, f_max=3.0)`, i.e. 2.23 → 2.53 Hz
   instead of 1.4 → 3.0. Duty 0.38 → 0.29, flight 56 → 83 ms per phase. The first value was the horse-scaled
   0.24 s (swing ~0.345 s × 0.69, duty 0.47 → 0.39, flight 16 → 43 ms): the gaits_v4 trot tracked it (88% of the
   scheduled flight airborne at 2.8 m/s) but it read as a grounded trot, so the air time was raised.
2. **Diagonal synchrony:** `touchdown_timing` with std 0.045 and weight 1.5 (the per-foot `foot_gait` cannot tell
   a late diagonal partner from a wrong foot), and `TROT.success_timing_tol = 0.06` (2 s.d. of the horse
   ratios) so SGS only counts strides that are real trots. The footfall-spacing metric is the one the walk
   uses; for the trot all four expected spacings are 0.5.
3. **At most two feet down:** `excess_contact`, −0.3 per foot beyond two. Stops a 4-beat walk-trot from
   collecting most of the per-foot match reward.
4. **Speed-independent clearance:** `feet_clearance(speed_normalized=True)`. Swing-foot speed is ~2× body speed,
   so the unnormalised cost grew ~2.3× from 1.2 to 2.8 m/s. The weight is rescaled to −1.5. (The trot task's
   `feet_swing_height`, peak height at landing, was not ported.)
5. **Relative tracking kernel:** `track_linear_velocity(std_rel=0.2)`, width max(0.5, 0.2·v_cmd). The walk keeps
   `std_rel = 0` (fixed width).
6. **No fore/hind asymmetry:** the horse trot's hind stance is only 1.6% shorter than the fore, so the clock
   stays symmetric (no change).

Not yet tuned: these weights are first values from the trot task and the horse data, and nothing has been
trained with them.

### Canter (`rewards/canter.py`)

Starts from the trot's term set and changes what the canter does differently. It replaces `common_rewards`.
Horse references: Robilliard, Pfau & Wilson 2007 (`187.pdf`, footfall timing) and St. George et al. 2023
(`animals-13-01755.pdf`, leading vs trailing limbs). The clock is `gaits.CANTER`.

| Term | Weight | Function | Notes vs the trot |
|---|---|---|---|
| `track_linear_velocity` | +1.0 | `track_linear_velocity` | std 0.8 (`CANTER.tracking_std`), width max(0.8, 0.2·v_cmd), z weight **0.3** (trot 0.5) |
| `track_angular_velocity` | +1.0 | `track_angular_velocity` | same |
| `track_yaw_fine` | +0.5 | `track_angular_velocity` | std 0.25, `xy_weight` 0 (shared, `rewards/common.py`) |
| `track_turn_lean` | +0.5 | `track_turn_lean` | exp(−(g_y − sin θ)²/0.07²), θ = atan(v²κ/g); shared, `rewards/common.py` |
| `foot_gait` | **+1.5** (first 0.5) | `gait_contact_schedule` | **exact_weight 0.5**: half per-foot, half all-four-feet match. At 0.5 per-foot only the gaits_v4 canter drifted to foot match 0.73, all-four 0.24, 47% flight vs 15–22% scheduled |
| `touchdown_timing` | **+3.0** (first 1.5) | `touchdown_timing` | std **0.05** (trot 0.045): horse canter ratios are 0.27 ± 0.03 fore, ± 0.05 hind |
| `unscheduled_flight` | −1.0 | `unscheduled_flight` | **new** (the gallop's): all feet up outside the clock's flight window |
| `diagonal_order` | −1.0 | `diagonal_order` | **new.** 1 on the step RR lands while FL is not down yet |
| `excess_contact` | −0.3 | `feet_excess_contact` | `max_feet=3` (trot 2): a canter has 0–3 feet down |
| `foot_suspension` | **+2.0** (first 0.5) | `feet_suspension` | as the trot. One flight per cycle, 0.15–0.22 of it with the compressed clock (0.02–0.08 before) |
| `body_orientation_l2` | −1.0 | `body_orientation_roll_pitch` | roll 1.0, pitch **0.2** (trot 1.0 / 1.0) |
| `body_ang_vel` | −0.05 | `body_angular_velocity_roll_pitch` | roll 1.0, pitch **0.2** |
| `angular_momentum` | −0.025 | `angular_momentum_penalty` | same |
| `pose` | +1.0 | `variable_posture` | trot's stds, "running" at every canter command |
| `foot_clearance` | −1.5 | `feet_clearance` | target **0.16 m** (trot 0.15), speed-normalised |
| `foot_slip` | −0.25 | `feet_slip` | same |
| `soft_landing` | −1e-3 | `soft_landing` | same |
| `is_terminated`, `joint_acc_l2`, `joint_pos_limits` | −200, −2.5e-7, −10 | mjlab | same |
| `is_alive` | +0.75 | mjlab | per second alive; forked gaits only (see Fork) |
| `action_rate_l2`, `torque_l2` | −0.2, −1e-5 | mjlab | same |

`stand_still` is dropped (the canter is never commanded to stand).

#### Horse-data basis

- **Footfall order (187, Table 1).** Left canter is RH, RF before or at the same time as LH, LF; a gallop is RH, LH,
  RF, LF. The only difference is whether the diagonal fore lands before its hind. `touchdown_timing` cannot see
  that (FL and RR are scheduled 0.01 of a cycle apart), so `diagonal_order` checks it on the contact sensor.
  Simultaneous landings are allowed. Logged as `Metrics/diagonal_order_violations`.
- **Footfall spacing (187, Table 4, right canter).** Fore ratio (RF − LF)/T = 0.27 ± 0.03, hind 0.67 ± 0.05
  measured as |RH − LH| from LF, i.e. 0.33 as RR − RL. The horse-exact clock `(FR 0.60, FL 0.32, RR 0.33, RL 0)`
  is **compressed ×0.85** to `(FR 0.51, FL 0.272, RR 0.281, RL 0)`: FR − FL = 0.24, RR − RL = 0.28 (~1 s.d. low),
  same order, so the stance sequence ends earlier and leaves room for a visible flight (see Flight). The SGS timing test (`success_timing_tol` 0.08, ~2 s.d.) uses the existing
  FL-to-FL stride metric unchanged; its expected spacings come from this clock (0.28, 0.33, 0.32, 0.27).
- **Fixed air time (187, Table 3).** Horse swing stays at ~285–320 ms from 5 to 9 m/s while stance shrinks
  (184 → 122 ms), so as for the trot `GaitClock(air_time=0.20)` (0.30 s × 0.69) and the duty follows the stride
  frequency. Horse stride frequency is roughly 2.0 → 2.5 Hz; scaled ×1.44 and fitted: `f = 2.04 + 0.25 v`.
  2.74 Hz at 2.8 m/s against the trot's 2.53 Hz at the top of its range.
- **Flight.** One flight after FR lifts, 0.49 − duty of the cycle (`air_time` 0.22 s, duty 0.34 → 0.272): 15% /
  58 ms at 2.24 m/s up to 22% / 62 ms at 6 m/s. The horse-exact clock gave 0.40 − duty with duty 0.38 → 0.32:
  only 8 → 23 ms, the short flight that made the first gaits_v4 trot look grounded. **`duty_min` = the FL
  touchdown (0.272) matters:** a duty below it lifts RL before the diagonal pair lands and the stride gets a
  second flight phase, which is not a canter (checked numerically: one flight phase, 0–3 feet down, at every
  speed). At most
  3 feet are ever down at every speed.
- **Pitch is part of the gait (St. George 2023).** The trunk pitches nose-down through the trailing-fore +
  leading-hind diagonal support (~50% of the stride) and back up through lead-fore support and flight, so pitch
  is weighted 0.2 against roll 1.0 and the vertical-velocity weight in the tracking kernel is lowered.
- **No per-leg stance asymmetry.** The two papers disagree on which fore limb has the longer stance (St. George:
  trailing 0.29 s vs leading 0.27 s; Robilliard: lead limbs slightly longer) and the difference is ≤ 7%, so
  all legs share one duty factor.

Open points:
- The clock is right lead only. A left lead needs a mirrored clock and a lead flag in the observations,
  which changes the observation layout, so decide before the first canter checkpoint is trained.
- 2.24 m/s is Froude 0.88, a horse at ~3.2 m/s, below the slowest canter in either paper (4 m/s). The low end
  of the range is extrapolated.
- Not implemented: a leading-limb protraction term (St. George: lead limbs reach further forward in swing). Try
  it only if the gait comes out symmetric once the timing terms work.
- Not tuned: these weights are first values from the trot and the horse data; nothing has been trained with them.


### Gallop (`rewards/gallop.py`)

**Trained from scratch** (`StageBudget(fork_from_previous=False)`), not forked: it runs at 100 Hz, and the 50 Hz
canter actor forked into it fell within 3 s at every speed. It still runs after the canter and still starts 20% of
episodes from the `canter-up` bank. A random std-1.0 policy netted about −15 per second alive (action rate −9.2,
foot slip −3.6, joint acc −3.5) against −4 for terminating, so it starts at **action std 0.4**
(`scratch_init_std`) and **`mdp.PenaltyRamp`** scales `action_rate_l2`, `joint_acc_l2`, `foot_slip` and
`torque_l2` from 10% to full weight over the first 500 iterations (`penalty_ramp_iterations`; logged as
`Curriculum/penalty_ramp/scale`). Measured at the start: about +2.2 per second alive.

Starts from the canter's term set and changes what the gallop does differently. It replaces `common_rewards`.
Horse references: Witte, Hirst & Wilson 2006 (`4389.pdf`, stride timing of 9 Thoroughbreds at 9–17 m/s, 5642
strides) and Self Davies, Spence & Wilson 2019 (`12348_Ground-reaction-forces-...pdf`, force plates under 7
galloping Thoroughbreds at 9–14 m/s). The clock is `gaits.GALLOP`. Runs at **100 Hz**.

| Term | Weight (100 Hz) | Function | Notes vs the canter |
|---|---|---|---|
| `track_linear_velocity` | +1.0 | `track_linear_velocity` | std 1.0, width max(1.0, 0.2·v_cmd), z weight **0.2** (canter 0.3) |
| `track_angular_velocity` | +1.0 | `track_angular_velocity` | same |
| `track_yaw_fine` | +0.5 | `track_angular_velocity` | std 0.25, `xy_weight` 0 (shared, `rewards/common.py`) |
| `track_turn_lean` | +0.5 | `track_turn_lean` | exp(−(g_y − sin θ)²/0.07²), θ = atan(v²κ/g); shared, `rewards/common.py` |
| `foot_gait` | **+1.5** (first 0.5) | `gait_contact_schedule` | exact_weight 0.5 as the canter (per-leg duty now) |
| `touchdown_timing` | +4.0 (2.0 × 2) | `touchdown_timing` | std **0.05** cycle ≈ 1.5 steps at 100 Hz |
| `footfall_sequence` | +1.0 (0.5 × 2) | `footfall_sequence` | **new.** 1 at the FR touchdown when RL < RR < FL < FR landed in that strict order within the stride. Replaces `diagonal_order` |
| `excess_contact` | −0.3 | `feet_excess_contact` | `max_feet=2` (canter 3) |
| `foot_suspension` | +1.0 | `feet_suspension` | canter 0.5: the flight is ~7–8 steps now, not < 1 |
| `unscheduled_flight` | −1.0 | `unscheduled_flight` | **new.** 1 while all feet are up outside the clock's flight window |
| `flight_duration` | +1.0 (0.5 × 2) | `flight_duration` | **new.** At the RL touchdown that ends a flight: exp(−((t − t_sched)/0.025 s)²) |
| `grf_peak_excess` | −0.5 | `grf_peak_excess` | **new**, replaces `soft_landing`. Σ relu(Fz_peak / Fz_pred − 1.3)² |
| `vertical_impulse_balance` | +0.6 (0.3 × 2) | `vertical_impulse_balance` | **new.** Per stride, exp(−((fore share − target)/0.05)²) |
| `propulsion_split` | −0.4 (0.2 × 2) | `propulsion_split` | **new.** Per stride, (relu(−Ix_hind) + relu(Ix_fore)) / (m g T) |
| `body_orientation_l2` | −1.0 | `body_orientation_roll_pitch` | roll 1.0, pitch **0.1** (canter 0.2) |
| `body_ang_vel` | −0.05 | `body_angular_velocity_roll_pitch` | roll 1.0, pitch **0.1** |
| `angular_momentum` | −0.025 | `angular_momentum_penalty` | same |
| `pose` | +1.0 | `variable_posture` | same |
| `foot_clearance` | −1.5 | `feet_clearance` | same 0.16 m (neither paper has swing height) |
| `foot_slip` | −0.25 | `feet_slip` | same |
| `is_terminated` | −400 (−200 × 2) | mjlab | same per termination |
| `is_alive` | +0.75 | mjlab | per second alive; forked gaits only (see Fork) |
| `joint_acc_l2`, `joint_pos_limits`, `torque_l2` | −2.5e-7, −10, −1e-5 | mjlab | same |
| `action_rate_l2` | −0.4 (−0.1 × 4) | mjlab | = −0.1 at 50 Hz, half the canter's: stances need fast joint motion |

**Weights at 100 Hz.** mjlab pays `weight · term · dt`. Dense terms are per second and keep their weights. Sparse
once-per-event terms (touchdowns, strides, terminations) would be worth half as much per event at 100 Hz, so their
weights are ×2 (`REFERENCE_DT / control_dt`, written as `w * event` in the module). `action_rate_l2` is a squared
per-step difference: for the same smooth motion it is 4× smaller per step, so its weight is ×4.

#### Horse-data basis

Scaling: Froude similarity with a 1.5 m horse leg (Witte's estimate) against 0.58 m here. Speeds and frequencies
×√(1.5/0.58) ≈ 1.61, times ×0.62. 4.47–8 m/s here is a horse at ~7.2–12.9 m/s. Horse fits below are Witte's
quadratic population fits (Figs 3–9) or Self Davies' linear fits (Figs 2–3), with v the horse speed.

| | Horse | Robot clock, 4.47 → 8 m/s |
|---|---|---|
| Stride frequency | 2.02 → 2.41 Hz over 9–17 m/s | `f = 2.54 + 0.127 v` = 3.11 → 3.56 Hz (canter 3.16 Hz at 4.47: continuous handoff) |
| Swing (protraction) | fore 364 → 342 ms, hind 355 → 326 ms | air time fore 0.225 s, hind 0.217 s |
| Duty | fore 0.27 → 0.18, hind 0.29 → 0.22 (hind always longer) | fore 0.30 → 0.20, hind 0.33 → 0.23 |
| Flight | ~27% of the stride at every speed, 135 → 119 ms, never two per stride | one flight, 64 → 84 ms (horse ×0.62: 84 → 74 ms) |
| Overlap between footfalls | 183 → 35 ms in total, falls toward 0 | 50 → 15 ms per transition, always > 0, ≤ 2 feet down |
| Stance length | fore 1.18 → 1.31 m, hind 1.28 → 1.58 m (sweep 23–32°) | v·duty/f: fore 0.43 → 0.45 m, hind 0.47 → 0.51 m |
| Fore share of vertical impulse | 49:51 mean; hind % = 36.8 + 1.215 v | 0.632 − 0.0195 v = 0.545 → 0.476 (robot standing: 0.58) |
| Peak vertical GRF | 1.0–1.7 BW; duty-factor prediction = 100% of recorded (middle half 90–107%) | prediction π P m g / (4 duty): 1.4 → 1.9 BW fore, 1.1 → 1.8 BW hind |
| Fore-aft impulse | hind net +0.08 / +0.04, fore 0 / −0.03 N s/kg | hind ≥ 0, fore ≤ 0 |
| Lead vs non-lead | no significant timing difference | identical clock within each pair |

- **The previous placeholder clock had two flights** (RR lifted at 0.40, FL landed at 0.45) and a 0.15-cycle main
  flight. Its frequency `f = v · 0.28 / 0.5` was 2.5 Hz at 4.47 m/s (below the canter) and hit the 4 Hz cap by
  7.1 m/s, so speed came from cadence instead of stride length. Replaced.
- **Clock checked numerically** (2000-sample grid, 4.47 → 8 m/s): exactly one flight segment, at most 2 feet
  down, all three footfall overlaps positive. With FR at 0.49 there were 3 feet down for 2 ms at 4.47 m/s, hence 0.50.
- **Why `footfall_sequence` and not `diagonal_order`.** `diagonal_order` asks the partner foot to still be
  down. At 8 m/s FL lifts ~13 ms before FR lands, which is correct, so it would fire wrongly. The sequence test
  uses touchdown times instead and is strict (two feet on one step fail it).
- **Force terms.** The foot contact sensor (`netforce`, world frame) reports z < 0, the force on the ground. The
  terms flip the sign from z and integrate every physics substep (`history_length = decimation`). Standing, the
  summed GRF is 1.000 × body weight. Strides run from one RL touchdown to the next; partial strides
  (first after reset, or outside 0.5–1.5 cycles) are skipped. Fore-aft is in the heading (yaw) frame.
- **Verified on a synthetic perfect gallop** (feet on the clock, half-sine GRFs): `footfall_sequence` 1 every
  stride; `flight_duration` 75 ms against a 74 ms target; measured fore share = injected; stride impulse
  0.999 BW; no penalties. With canter order the sequence term pays 0. With reversed push/brake
  `propulsion_split` costs 0.157 per stride. With 2× forces `grf_peak_excess` fires.

New metrics: `flight_time_ms` / `flight_target_ms`, `footfall_sequence_ok`, `unscheduled_flight`,
`fore_impulse_share` / `fore_impulse_target`, `stride_impulse_bw` (should be ~1), `hind_net_impulse_x` /
`fore_net_impulse_x` (N s/kg), `grf_peak_bw`, `grf_peak_over_pred` (mean over loaded foot-steps; a perfect
half-sine reads ~0.7).

Open points:
- Right lead only. A mirrored clock in refine would give a left lead. The per-leg sin/cos clock already shows
  which lead is active, so the obs layout would not change. But the canter (and its handoff banks) is
  right-lead, and the success test's stride metric is FL-to-FL.
- Not tuned: first values. Nothing has been trained with them.
- 400 Nm sim limit: ~600 N peak foot force is fine there, probably not with the real 120 Nm motors. Watch
  `Metrics/peak_torque`.
- The 50 Hz canter → 100 Hz gallop fork is untested. If the forked actor collapses, try more critic warmup
  iterations or a higher std floor for the gallop fork.

## Turns (refine only)

The refine stage of every gait trains turns. Handoff stays straight, and the handoff banks are
recorded straight.

- **Non-holonomic arcs.** `ang_z = sign · lin_x_cmd / r`. The robot never turns on the spot. r and the direction
  (±, 50/50) are fixed for the episode. ang_z is recomputed from the commanded speed whenever that changes
  (mid-episode resamples). Moving spawns start on the arc (yaw rate = ang_z).
- **Radius is an SGS difficulty axis.** 9 radius bins, log-spaced from 2000 m (straight in practice) down to
  `GaitSpec.turn_radius_min`. Configs are speed bin × radius bin. Each speed bin has its own **radius frontier**,
  which starts at 2000 m and moves one bin tighter when that (speed, radius) config reaches ≥ 0.8 success over
  ≥ 50 outcomes. 50% of episodes are drawn at a frontier config (uniform speed bin); the
  rest are SGS Beta-kernel over every unlocked config (radius at or wider than the frontier). A mid-episode
  speed resample stays among the speed bins that have the episode's radius unlocked.
- **Feasibility.** A config is used only if v_hi² / r ≤ `turn_max_lat_accel` = 0.4 g. Foot friction is
  randomised at 0.6–1.2, so this stays under the grip limit. Tightest radius reached per gait:

  | Gait | R_min | Tightest feasible radius by speed | Max yaw rate |
  |---|---|---|---|
  | walk | 1.0 m | 1 m at every speed (90/90 configs) | 1.3 rad/s |
  | trot | 1.5 m | 1.5 m below ~2.34 m/s, 3.7 m above (122/126) | 1.6 rad/s |
  | canter | 2.5 m | 2.5 m → 5.8 m (from ~3.0 m/s) → 13.3 m (from ~4.75 m/s) (118/135) | 1.2 rad/s |
  | gallop | 4.0 m | 8.7 m below ~5.7 m/s, 18.9 m above (103/126 configs) | 0.66 rad/s |

  Radii (log steps of ~2.6×), walk: 2000, 773, 299, 116, 45, 17, 6.7, 2.6, 1.0 m. The first four or five bins are
  straight in practice (yaw rate ≤ 0.03 rad/s at 1.3 m/s), so they pass quickly. The bins are coarse, so the
  tightest feasible radius jumps. 4 m is never feasible for the gallop.
- **Success test.** Speed and termination as before, plus the yaw test on the signed yaw-rate error:
  `|mean(ω − ω_cmd)| ≤ max(0.2 rad/s, 0.2 · mean|ω_cmd|)`. The footfall-timing test is **off** for every gait.
  In `gaits_v2` the walk refine's `footfall_timing_error` stayed at about 0.26, against a 0.05 tolerance. So
  every episode failed, the radius frontier never left the widest radius, and the policy never saw a turn
  (SGS state of `walk-refine/model_4999.pt`: 0% success in all 90 configs, frontier 0 in every speed bin).
- **Rewards (all gaits, shared functions).** `body_orientation_roll_pitch` takes `command_name` and measures roll
  from the turn's equilibrium lean `θ = atan(v² κ / g)` (the walking task's term; upright when straight).
  `angular_momentum_penalty` takes `command_name` and drops L_z on turning envs: a commanded yaw rate needs yaw
  angular momentum, and `track_angular_velocity` already scores yaw. `track_angular_velocity` tracks
  `cmd[:, 2]`. **`track_yaw_fine`** (+0.5, every gait, `rewards/common.py`) is a second, sharper yaw term:
  `track_angular_velocity` with std 0.25 rad/s and no roll/pitch part (`xy_weight` 0). The main term's kernel
  barely separates turning from not turning: ignoring a 0.2 rad/s command still scores 0.85 (walk) or 0.92
  (others) of it, against 0.53 for this term. **`track_turn_lean`** (+0.5, every gait) rewards the lean itself:
  `exp(−(g_y − sin θ)² / 0.07²)` on `base_link`, with the same target θ as `body_orientation_roll_pitch`. That L2
  term is flat near its minimum: staying upright through an 8.4° turn (walk, 1.2 m/s on 1 m) costs only 0.021
  per step, and the gaits_v4 walk did not lean at all (< 2° everywhere). This kernel scores upright 0.01 there.
  Targets: 2.9° (0.7 m/s, 1 m), 8.4° (1.2 m/s, 1 m), 12° (trot 2.8 m/s, 3.7 m), 15–19° (canter / gallop at
  the 0.4 g cap). Nothing gait-specific was needed. Canter and gallop are right-lead only, so their left turns run
  on the counter-lead.
- **Observation layout change.** `gait_command` gained ω_cmd: actor 57 → 58, critic 88 → 89. Checkpoints from
  before this change (e.g. run `gaits_v1`) no longer load.

## Metrics (logged, never rewarded)

- `gait_foot_match`, `gait_exact_match`
- measured `duty_factor`, `flight_phase`
- `forward_speed_error`
- `footfall_timing_error` (walk: mean worst footfall-spacing deviation per stride)
- `peak_torque`, `mechanical_power`
- SGS: `v_current`, `frontier_bin`, `frontier_success`, `episode_success`; refine also `sampled_radius` (median),
  `sampled_yaw_rate`, `radius_frontier_mean`, `tightest_radius_unlocked`

## Run layout

```
scripts/logs/rsl_rl/multigait/<run>/
  pipeline.json        plan, per-job status/results, deployed policies (resume state)
  results/<job>.json   checkpoints + log dir of each training stage
  banks/<job>.pt       handoff banks
  job_logs/<job>.log   stdout of each job
  <job>/<timestamp>_<job>/model_*.pt, policy.onnx
```
