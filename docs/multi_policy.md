# Multi-gait pipeline (`centaur_rl/tasks/Multi_Policy`)

One command trains four gait policies in sequence: walk → trot → canter → gallop. The
command is a straight line on flat ground (`lin_x` only, never `lin_y` or `ang_z`).
Each gait gets its own policy. Every gait after the walk starts from the previous gait's
weights.

```bash
cd scripts
python train.py --multi-Gait_straight_flat_SGS --env.scene.num-envs 4096          # everything
python train.py --task Multi-Gait_straight_flat_SGS --dry-run                     # print the plan
python train.py --task Multi-Gait_straight_flat_SGS --run-name R --resume         # continue run R
python train.py --task Multi-Gait_straight_flat_SGS --gaits walk trot --handoff-iterations 0
python train.py --task Multi-Gait_straight_flat_SGS-Trot-Ladder --agent.stage.init-mode fork \
    --agent.stage.init-checkpoint <walk ladder model_N.pt>                         # one stage by hand
python play.py Multi-Gait_straight_flat_SGS-Trot-Refine --checkpoint-file <model.pt>
```

Unknown options are forwarded to every training stage (`--env.*`, `--agent.*`, `--gpu-ids`, ...).

## Stages per gait

| Stage | What it trains | Starts from |
|---|---|---|
| `g-ladder` | SGS speed ladder: 80% of episodes at the frontier speed **V_current**, 20% replay of slower mastered bins. The frontier moves up one bin when its success rate reaches 0.8 (at least 50 outcomes). Command fixed per episode. | walk: from scratch. Other gaits: **fork** of the previous gait's ladder checkpoint (`fork_iteration`, default its last one) |
| `g-refine` | whole speed range, SGS-weighted, command resampled every 4–8 s, up and down. 20% of resets start from the slower gait's top-speed states. | **continue** from `g-ladder` |
| `g-up` / `g-down` | record handoff banks: states of `g-refine` at the gait's top or lowest speed | — |
| `g-handoff` (optional, 500 it) | as refine, with 50% of resets from neighbour banks (slower gait's `up`, faster gait's `down`) | continue from `g-refine` |

**Fork** (new gait):
- The actor weights and the actor's observation normalizer are loaded; the critic and optimizer start fresh.
- The actor is frozen for the first 200 iterations, so only the critic trains.
- Action std is raised to at least 0.6.
- The actor normalizer's sample count is capped at 2M, so its mean and std re-adapt to the new gait. It is never frozen.

**Continue** (same gait): actor, critic and optimizer are all loaded.

Default budgets are in `gaits.StageBudget`: ladder 1000, refine 10000, handoff 500, critic warmup 200.

Each gait's deployed policy is the final checkpoint of `g-handoff`, or of `g-refine` if
handoff is disabled. The paths are written to `pipeline.json` → `deployed_policies`.

## Gaits (`gaits.py`)

| Gait | Speed (m/s) | Clock (touchdown FR, FL, RR, RL / duty) | Spawn |
|---|---|---|---|
| walk | 0–1.30 (SGS from 0.15) | .476, .976, .226, .726 (lateral sequence, footfalls 0.25 apart) / 0.79 → 0.57 with speed | at rest |
| trot | 1.20–2.80 | 0, .5, .5, 0 (diagonal pairs) / 0.47 → 0.39 with speed | moving at the command |
| canter | 2.24–6.00 | 0.60, .32, .33, 0 (right lead: RL, FL → RR, FR) / 0.38 → 0.32 with speed | moving |
| gallop | 4.47–8.00 (top end is a placeholder) | .57, .45, .12, 0 / 0.28 | moving |

Walk/trot, trot/canter and canter/gallop all overlap (`check_gait_chain`). The walk tops out at Froude 0.30
and the trot starts at 0.25, where horses change gait (Robilliard et al. 2007, 187.pdf).

Stride frequency rises with speed: `f(v) = clamp(v · duty / stance_sweep, f_min, f_max)` for the
gallop. The walk uses `f(v) = clamp(0.7 + 0.95 v, 0.6, 1.7)` Hz (strides lengthen faster than they quicken, as in
horses), and its duty factor follows the frequency with a fixed air time per foot:
`duty = clamp(1 − 0.25 s · f, 0.55, 0.80)` (0.79 at 0.15 m/s, 0.57 at 1.3 m/s, always 2–3 feet down). Set per gait
by `GaitClock.air_time` and `StrideFrequency.slope/offset`; `gait_duty` is the per-env duty, so the observation
layout is unchanged. The trot does the same with `f = 2.0 + 0.19 v` (2.0–3.0 Hz) and `duty = clamp(1 − 0.24 s · f,
0.35, 0.50)` (0.47 at 1.2 m/s, 0.39 at 2.8 m/s). The canter uses `f = 2.04 + 0.25 v` (2.60–3.54 Hz) and
`duty = clamp(1 − 0.20 s · f, 0.32, 0.38)`.

Fixed for every gait, because changing any of these invalidates every checkpoint:
- the observation layout
- action scale 0.5
- kp 150 / kd 5
- torque limit 400 Nm (sim only)
- 50 Hz control

## Observations

Actor:
- base angular velocity
- projected gravity
- `gait_command` [v_cmd, f]
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

Each gait's own design goes in `rewards/<gait>.py`. The gallop is still a placeholder.

### Walk (`rewards/walk.py`)

The walking task's term set and weights (see [walking.md](walking.md)). It replaces `common_rewards`.

| Term | Weight | Function | Notes |
|---|---|---|---|
| `track_linear_velocity` | +1.0 | `track_linear_velocity` | std 0.35 (`WALK.tracking_std`; the walking task used 0.5) |
| `track_angular_velocity` | +1.0 | `track_angular_velocity` | std 0.5, holds w_z = 0 |
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

### Trot (`rewards/trot.py`)

The trot task's term set and weights (see [trot.md](trot.md)). It replaces `common_rewards`.

| Term | Weight | Function | Notes |
|---|---|---|---|
| `track_linear_velocity` | +1.0 | `track_linear_velocity` | kernel width max(0.5, 0.2·v_cmd) m/s (0.5 = `TROT.tracking_std` = the trot task's √0.25; widens above 2.5 m/s), z weight 0.5 (flight needs base bounce) |
| `track_angular_velocity` | +1.0 | `track_angular_velocity` | std √0.5, holds w_z = 0 |
| `foot_gait` | +0.5 | `gait_contact_schedule` | exact_weight 0: fraction of feet matching the clock (the trot task's `feet_gait`) |
| `touchdown_timing` | +1.5 | `touchdown_timing` | **new.** std 0.045 cycle: diagonal partners must land together (horse ratios 0.50 ± 0.03); sparse, 4 landings per cycle |
| `excess_contact` | −0.3 | `feet_excess_contact` | **new.** feet on the ground beyond 2 (a trot has 0–2) |
| `foot_suspension` | +0.3 | `feet_suspension` | 1 while the clock schedules a flight and no foot is down; that is 1 − 2·duty of the cycle, 0.06 at 1.2 m/s to 0.21 at 2.8 m/s |
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
| `action_rate_l2` | −0.2 | mjlab | 4× the trot task's −0.05: action scale 0.5 instead of 0.25 |

Differences from the trot task's own clock (fixed 0.6 s period, flight fraction 0.02→0.14 with speed): here
`foot_gait`, `foot_suspension`, `touchdown_timing` and `excess_contact` read `GaitCommand.stance_target`, so the
flight phases follow `TROT.clock`. Horse-based, it holds each foot's air time at 0.24 s and lets the duty factor
follow the stride frequency, so flight (1 − 2·duty)/(2f) grows with speed like the trot task's: 16 ms at
1.2 m/s, 30 ms at 2.0, 43 ms at 2.8.

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

1. **Duty falls with speed at a fixed air time** (`gaits.TROT`): `GaitClock(air_time=0.24, duty_min=0.35,
   duty_max=0.50)` (0.24 s = horse swing ~0.345 s × 0.69) and `StrideFrequency(slope=0.19, offset=2.0,
   f_min=2.0, f_max=3.0)`, i.e. 2.23 → 2.53 Hz instead of 1.4 → 3.0. Duty 0.47 → 0.39, flight 16 → 43 ms. The
   trot is almost flight-free at 1.2 m/s, which eases the handoff from the walk.
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
| `foot_gait` | +0.5 | `gait_contact_schedule` | same |
| `touchdown_timing` | +1.5 | `touchdown_timing` | std **0.05** (trot 0.045): horse canter ratios are 0.27 ± 0.03 fore, ± 0.05 hind |
| `diagonal_order` | −1.0 | `diagonal_order` | **new.** 1 on the step RR lands while FL is not down yet |
| `excess_contact` | −0.3 | `feet_excess_contact` | `max_feet=3` (trot 2): a canter has 0–3 feet down |
| `foot_suspension` | +0.5 | `feet_suspension` | trot 0.3. One flight per cycle, 0.02–0.08 of it, so a sparser signal |
| `body_orientation_l2` | −1.0 | `body_orientation_roll_pitch` | roll 1.0, pitch **0.2** (trot 1.0 / 1.0) |
| `body_ang_vel` | −0.05 | `body_angular_velocity_roll_pitch` | roll 1.0, pitch **0.2** |
| `angular_momentum` | −0.025 | `angular_momentum_penalty` | same |
| `pose` | +1.0 | `variable_posture` | trot's stds, "running" at every canter command |
| `foot_clearance` | −1.5 | `feet_clearance` | target **0.16 m** (trot 0.15), speed-normalised |
| `foot_slip` | −0.25 | `feet_slip` | same |
| `soft_landing` | −1e-3 | `soft_landing` | same |
| `is_terminated`, `joint_acc_l2`, `joint_pos_limits` | −200, −2.5e-7, −10 | mjlab | same |
| `action_rate_l2`, `torque_l2` | −0.2, −1e-5 | mjlab | same |

`stand_still` is dropped (the canter is never commanded to stand).

#### Horse-data basis

- **Footfall order (187, Table 1).** Left canter is RH, RF before or at the same time as LH, LF; a gallop is RH, LH,
  RF, LF. The only difference is whether the diagonal fore lands before its hind. `touchdown_timing` cannot see
  that (FL and RR are scheduled 0.01 of a cycle apart), so `diagonal_order` checks it on the contact sensor.
  Simultaneous landings are allowed. Logged as `Metrics/diagonal_order_violations`.
- **Footfall spacing (187, Table 4, right canter).** Fore ratio (RF − LF)/T = 0.27 ± 0.03, hind 0.67 ± 0.05
  measured as |RH − LH| from LF, i.e. 0.33 as RR − RL. Clock `touchdown = (FR 0.60, FL 0.32, RR 0.33, RL 0)`:
  FR − FL = 0.28, RR − RL = 0.33. The SGS timing test (`success_timing_tol` 0.08, ~2 s.d.) uses the existing
  FL-to-FL stride metric unchanged; its expected spacings come from this clock (0.28, 0.33, 0.32, 0.27).
- **Fixed air time (187, Table 3).** Horse swing stays at ~285–320 ms from 5 to 9 m/s while stance shrinks
  (184 → 122 ms), so as for the trot `GaitClock(air_time=0.20)` (0.30 s × 0.69) and the duty follows the stride
  frequency. Horse stride frequency is roughly 2.0 → 2.5 Hz; scaled ×1.44 and fitted: `f = 2.04 + 0.25 v`.
  2.74 Hz at 2.8 m/s against the trot's 2.53 Hz at the top of its range.
- **Flight.** One flight after FR lifts, 0.40 − duty of the cycle: 8 ms at 2.24 m/s (duty capped at 0.38) up to
  23 ms at 6 m/s (duty 0.32). Below 20 ms the flight is under one control step, so `foot_suspension` is
  mostly a bonus at the upper half of the speed range. **`duty_min = 0.32` matters:** the FL touchdown is at
  0.32, so a duty below it lifts RL before the diagonal pair lands and the stride gets a second flight phase,
  which is not a canter (checked numerically: with 0.28 the 6 m/s clock has two flight segments). At most
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


## Metrics (logged, never rewarded)

- `gait_foot_match`, `gait_exact_match`
- measured `duty_factor`, `flight_phase`
- `forward_speed_error`
- `footfall_timing_error` (walk: mean worst footfall-spacing deviation per stride)
- `peak_torque`, `mechanical_power`
- SGS: `v_current`, `frontier_bin`, `frontier_success`, `episode_success`

## Run layout

```
scripts/logs/rsl_rl/multigait/<run>/
  pipeline.json        plan, per-job status/results, deployed policies (resume state)
  results/<job>.json   checkpoints + log dir of each training stage
  banks/<job>.pt       handoff banks
  job_logs/<job>.log   stdout of each job
  <job>/<timestamp>_<job>/model_*.pt, policy.onnx
```
