# Velocity task (flying trot)

Task IDs: `Centaur-Rough`, `Centaur-Flat`, `Centaur-Rough-SGS`, `Centaur-Flat-SGS`.
Code: `centaur_rl/tasks/velocity/`. Shared structure, robot and reward functions: see [README.md](README.md).

The robot tracks a body-frame velocity command (vx, vy, yaw rate) with a **flying trot**: diagonal pairs
(FR+RL, FL+RR) alternate, and between the two stances all four feet are in the air.

## Command

| Axis | Range | First 5000 iterations |
|---|---|---|
| `lin_vel_x` | 1.0 – 2.0 m/s | 1.0 – 1.5 |
| `lin_vel_y` | ±0.5 m/s | ±0.3 |
| `ang_vel_z` | ±1.0 rad/s | ±1.0 |

No standing envs, no backward motion. Resampled every 3–8 s (`UniformVelocityCommand`).
Curricula: `terrain_levels_vel` (move up if the robot walked more than half a tile, down if less than half the
required distance) and `commands_vel` (stage widening at step 5000·24). SGS variants replace both.

## Gait schedule

Period 0.6 s, offsets `[FR 0, FL 0.5, RR 0.5, RL 0]`. A foot is in stance while its phase < 0.5 − flight(v), with
flight(v) = 0.02 + 0.06·‖cmd_xy‖ of the cycle (clamped at 2 m/s).

| ‖cmd_xy‖ | one flight phase | foot air time | stance |
|---|---|---|---|
| 1.0 m/s | 0.048 s | 0.348 s | 0.252 s |
| 1.5 m/s | 0.066 s | 0.366 s | 0.234 s |
| 2.0 m/s | 0.084 s | 0.384 s | 0.216 s |

## Rewards

| Term | Weight | Function | Notes |
|---|---|---|---|
| `track_linear_velocity` | +1.0 | `track_linear_velocity` | std √0.25, z weight 0.5 (flight needs base bounce) |
| `track_angular_velocity` | +1.0 | `track_angular_velocity` | std √0.5 |
| `body_orientation_l2` | −1.0 | `body_orientation_l2` | squared xy of projected gravity, `base_link` |
| `pose` | +1.0 | `variable_posture` | stds below; running threshold 1.0 so all commands use "running" |
| `body_ang_vel` | −0.05 | `body_angular_velocity_penalty` | |
| `angular_momentum` | −0.025 | `angular_momentum_penalty` | |
| `is_terminated` | −200 | mjlab | on failure terminations only |
| `joint_acc_l2` | −2.5e-7 | mjlab | |
| `joint_pos_limits` | −10 | mjlab | |
| `action_rate_l2` | −0.05 | mjlab | |
| `foot_gait` | +0.5 | `feet_gait` | fraction of feet matching the stance/swing schedule, when active |
| `foot_suspension` | +0.3 | `feet_suspension` | 1 while the schedule is in a flight phase and no foot touches the ground |
| `foot_clearance` | −1.0 | `feet_clearance` | target 0.15 m |
| `foot_slip` | −0.25 | `feet_slip` | |
| `soft_landing` | −1e-3 | `soft_landing` | |
| `stand_still` | −1.0 | `stand_still` | only below the command threshold (unused in practice: no standing) |

`pose` stds:

| Joint | standing | walking | running |
|---|---|---|---|
| `.*_1` | 0.05 | 0.15 | 0.15 |
| `.*_2` | 0.10 | 0.35 | 0.50 |
| `.*_3` | 0.30 | 0.80 | 1.00 |

### Task-specific functions (`mdp/rewards.py`)

- `gait_flight_fraction(speed, flight_fraction, flight_speed_range)`: linear speed → flight fraction map.
- `_gait_targets`: builds the per-foot scheduled stance (1/0) from period, offsets, threshold and flight fraction.
- `feet_gait`: mean match between actual contact and scheduled stance.
- `feet_suspension`: flight-phase indicator times "all feet airborne"; summed over a cycle it is the flight time
  achieved, so faster commands earn more.

## SGS variants

| | Rough-SGS | Flat-SGS |
|---|---|---|
| command axes | `lin_vel_x` [1, 2], `lin_vel_y` ±0.5, `ang_vel_z` ±1 | same |
| bins | 4 × 3 × 3 × 200 tiles = 7,200 configs | 8 × 5 × 7 = 280 configs |
| speed tolerance | max(0.3, 0.2·‖cmd‖) m/s | same |
| yaw tolerance | 0.3 rad/s | same |
| standing | none | none |
