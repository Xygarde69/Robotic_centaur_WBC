# Reward functions

Active reward terms for the Centaur velocity-tracking task. Defaults live in
`centaur_rl/tasks/velocity/velocity_env_cfg.py`; Centaur-specific overrides are in
`centaur_rl/tasks/velocity/config/centaur/env_cfgs.py`; implementations are in
`centaur_rl/tasks/velocity/mdp/rewards.py` (or mjlab built-ins where noted).

Total reward per step = sum over terms of `weight * term(env)`. Positive weight = reward,
negative weight = penalty. "Active" means the command magnitude
`‖cmd_xy‖ + |cmd_yaw|` exceeds `command_threshold` (0.1), i.e. the robot is asked to move.

## Active terms

| Term | Weight | Function | Key params |
|---|---|---|---|
| `track_linear_velocity` | +1.0 | `mdp.track_linear_velocity` | std = √0.25 |
| `track_angular_velocity` | +1.0 | `mdp.track_angular_velocity` | std = √0.5 |
| `body_orientation_l2` | −1.0 | `mdp.body_orientation_l2` | body = `base_link` |
| `pose` | +1.0 | `mdp.variable_posture` | per-joint stds below; walking thr. 0.1, running thr. 1.5 |
| `body_ang_vel` | −0.05 | `mdp.body_angular_velocity_penalty` | body = `base_link` |
| `angular_momentum` | −0.025 | `mdp.angular_momentum_penalty` | sensor `robot/root_angmom` |
| `is_terminated` | −200.0 | mjlab `is_terminated` | one-shot penalty on failure termination |
| `joint_acc_l2` | −2.5e-7 | mjlab `joint_acc_l2` | |
| `joint_pos_limits` | −10.0 | mjlab `joint_pos_limits` | |
| `action_rate_l2` | −0.05 | mjlab `action_rate_l2` | |
| `foot_gait` | +0.5 | `mdp.feet_gait` | period 0.6 s, stance threshold 0.56, trot offset `[0, 0.5, 0.5, 0]` |
| `foot_clearance` | −1.0 | `mdp.feet_clearance` | target height 0.15 m |
| `foot_slip` | −0.25 | `mdp.feet_slip` | sensor `feet_ground_contact` |
| `soft_landing` | −1e-3 | `mdp.soft_landing` | sensor `feet_ground_contact` |
| `stand_still` | −1.0 | `mdp.stand_still` | joints `.*` |

## Term descriptions

### Task (velocity tracking)
- **`track_linear_velocity`** — `exp(-(‖cmd_xy − v_xy‖² + 2·v_z²) / std²)`. Rewards matching the commanded
  body-frame xy velocity and penalizes vertical velocity. Range (0, 1].
- **`track_angular_velocity`** — `exp(-((cmd_yaw − ω_z)² + 0.05·‖ω_xy‖²) / std²)`. Rewards matching the
  commanded yaw rate, with a small penalty on roll/pitch rates. Range (0, 1].

### Posture / stability
- **`body_orientation_l2`** — squared xy of projected gravity in the `base_link` frame (0 when upright).
  Weight is negative, so tilt is penalized.
- **`pose`** (`variable_posture`) — `exp(-mean((q − q_default)² / std²))` with stds chosen by command speed
  (standing / walking / running). Tight when standing, loose when moving. Range (0, 1].
  Default pose is all joints 0 (straight legs). Knee stds are wide because every swing is a
  one-sided flexion away from straight (knees are one-way, horse-like `<>`).

  | Joint pattern | standing | walking | running |
  |---|---|---|---|
  | `.*_1` | 0.05 | 0.15 | 0.15 |
  | `.*_2` | 0.10 | 0.35 | 0.50 |
  | `.*_3` | 0.30 | 0.80 | 1.00 |
- **`body_ang_vel`** — squared roll/pitch angular velocity of `base_link` (yaw excluded).
- **`angular_momentum`** — squared magnitude of whole-body angular momentum.
- **`stand_still`** — sum of squared joint deviation from default, applied only when the command is
  *below* the threshold (robot told to stand).

### Gait
- **`foot_gait`** — fraction of feet whose contact state matches a periodic stance/swing schedule
  (per-foot phase offset; trot: FR+RL in phase, FL+RR half a period later). Only when active.
- **`foot_clearance`** — `Σ |z_foot − 0.15| · ‖v_foot,xy‖`; penalizes feet moving fast at the wrong
  height, encouraging a consistent swing height. Only when active.
- **`foot_slip`** — `Σ ‖v_foot,xy‖²` over feet in contact. Only when active.
- **`soft_landing`** — sum of contact-force magnitudes at the moment of first foot contact. Only when active.

### Regularization / safety
- **`joint_acc_l2`** — squared joint accelerations (smoothness).
- **`joint_pos_limits`** — penalty for joints beyond their soft position limits.
- **`action_rate_l2`** — squared change in action between steps.
- **`is_terminated`** — fires on failure terminations (not time-outs).

## Defined but not used
These exist in `mdp/rewards.py` but are not in the reward config:
- `self_collision_cost` — counts contact-force hits above a threshold.
- `feet_air_time` — rewards single-stance air time near a target.
- `feet_swing_height` — class; penalizes peak swing height error at landing.

## SGS tasks
The `*-SGS` tasks (`Centaur-Rough-SGS`) use the same reward set. SGS only changes what is sampled
(terrain tile, spawn, command), not the rewards. Its separate "success" signal (timeout without
failure, mean xy and yaw tracking error < 0.3) is used for sampling and is not a reward term.

## Walking tasks (`Centaur-Walking-*`)
`centaur_rl/tasks/walking/` starts from the reward set above and is tuned so the contact
schedule in `centaur_rl/tasks/walking/mdp/gait.json` is the dominant objective. Everything below
is derived from the gait file in `config/centaur/env_cfgs.py`, so swapping the JSON rescales it.

Straight-line walking only: the command is `lin_x` in (0, v_max), `lin_y = ang_z = 0`, plus a
gait cycle time as a 4th command entry (observed by the policy).

| Item | Velocity task | Walking task (walk, gait.json shape, duty 0.524) |
|---|---|---|
| `foot_gait` | +0.5, `feet_gait` (phase offsets) | +1.5, `feet_gait_clock`: 0.5 · per-foot match + 0.5 · [all 4 match], walk clock, startup from rest |
| `foot_contact_in_swing` | – | −0.5, fraction of feet on the ground while their clock says swing |
| `foot_clearance` | −1.0, target 0.15 m | −0.3, target 0.13 m |
| `body_orientation_l2` | −1.0, roll² + pitch² | −1.0, 0.3 · roll² + pitch² (`body_orientation_roll_pitch`) |
| `body_ang_vel` | −0.05, world x/y rates | −0.05, body-frame 0.3 · roll rate² + pitch rate² |
| command `lin_vel_x` | (−1, 2) | (0, min(2, v_max)) = (0, 1.687) |
| command `lin_vel_y`, `ang_vel_z` | (−1, 1) | 0 |
| command cycle time | – | [0.509, 1.66] s, feasible range for the speed (faster → shorter) |
| `track_linear_velocity` std | 0.5 | 0.5 |
| `track_angular_velocity` std | 0.707 | 0.25 (yaw command is always 0) |
| `pose` hip ab/ad std (walking, running) | 0.15 | 0.4 |
| `push_robot` velocities | ±0.5 m/s etc. | same (× min(1, 2·v_max)) |
| SGS success thresholds (xy, yaw) | 0.3, 0.3 | 0.3, 0.2 |
| SGS config | (tile, spawn, command) | command = (lin_x, 0, 0, cycle time) |
| `phase` observation | sin/cos, 0.6 s period | `walk_phase`: sin/cos of the walk clock + first-cycle progress |

- **Max walk speed** `v_max = sqrt(Fr · g · L)` with Froude number 0.5 (walk–trot transition)
  and leg length L = 0.58 m → 1.687 m/s.
- **Cycle time.** A planted foot sweeps `v · duty · T` under its hip; that stride is kept in
  [0.15, 0.45] m, so `T ∈ [0.15 / (duty·v), 0.45 / (duty·v)]` clipped to [0.509, 1.66] s.
  0.509 s is the cycle that reaches v_max with a full 0.45 m stride (0.24 s swing); 1.66 s is
  gait.json's own cycle. The 1.66 s cycle is allowed from 0.17 to 0.52 m/s and forced below 0.17 m/s, where strides drop under 0.15 m. All gait.json timings
  scale with T (shape and duty factor are fixed).
- **Gait clock** (`WalkVelocityCommand`): counts cycles since walking started, zero while
  standing, restarted at every episode reset and standing → walking switch.
- **`foot_gait`** (`feet_gait_clock`): as the trot, the fraction of feet whose contact matches
  their clock; stance while `(cycles + offset) % 1 < duty`. From rest each leg stays down
  until its first lift-off (FR 0, RL 0.25, FL 0.5, RR 0.75 of a cycle), so the walk starts with
  FR's long stride exactly as gait.json P0; afterwards P8 ≡ P0. Zero while standing.
- Offsets and lift phases are given in `LEG_NAMES` order and permuted to the contact sensor's
  foot order (model order FL, FR, RL, RR) by `feet_gait_clock`.
