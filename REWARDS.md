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
| `foot_clearance` | −1.0 | `mdp.feet_clearance` | target height 0.10 m |
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

  | Joint pattern | standing | walking | running |
  |---|---|---|---|
  | `.*_1` | 0.05 | 0.15 | 0.15 |
  | `.*_2` | 0.10 | 0.35 | 0.35 |
  | `.*_3` | 0.15 | 0.50 | 0.50 |
- **`body_ang_vel`** — squared roll/pitch angular velocity of `base_link` (yaw excluded).
- **`angular_momentum`** — squared magnitude of whole-body angular momentum.
- **`stand_still`** — sum of squared joint deviation from default, applied only when the command is
  *below* the threshold (robot told to stand).

### Gait
- **`foot_gait`** — fraction of feet whose contact state matches a periodic stance/swing schedule
  (per-foot phase offset; trot: FR+RL in phase, FL+RR half a period later). Only when active.
- **`foot_clearance`** — `Σ |z_foot − 0.10| · ‖v_foot,xy‖`; penalizes feet moving fast at the wrong
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
