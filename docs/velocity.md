# Velocity task (Unitree)

Task IDs: `Centaur-Velocity-Rough`, `Centaur-Velocity-Flat`.
Code: `centaur_rl/tasks/velocity/`. Shared structure, robot and reward functions: see [README.md](README.md).

A verbatim copy of the Go2 velocity task from `unitree_rl_mjlab` (`src/tasks/velocity`) running on the Centaur
robot, kept as a baseline. The rewards, commands, curricula and observations are upstream's, with no gait-specific
tuning. The flying-trot variant of this task is [trot.md](trot.md).

## What changed from upstream

| Item | Unitree Go2 | Centaur copy |
|---|---|---|
| robot | `get_go2_robot_cfg()` | `get_centaur_robot_cfg()` |
| joint patterns for `pose` | `*_hip_joint`, `*_thigh_joint`, `*_calf_joint` | `.*_1`, `.*_2`, `.*_3` (same std values) |
| foot geoms | `<leg>_foot_collision` | `FOOT_GEOM_NAMES` (same names) |
| termination | `illegal_contact` on non-foot geoms | `base_too_low` (base < 0.30 m above feet): only foot spheres collide |
| experiment name | `go2_velocity` | `centaur_velocity` |
| import paths | `src.tasks.velocity` | `centaur_rl.tasks.velocity` |

`mdp/terminations.py` gained `base_too_low`; nothing else in `mdp/` or `velocity_env_cfg.py` was edited.
Not tuned for Centaur: the `pose` stds are Go2's, so the knee std (0.15 standing, 0.5 moving) is tighter than the
trot task's (0.3 / 0.8–1.0) and may penalise Centaur's larger knee flexion.

## Command

`UniformVelocityCommand`, resampled every 3–8 s, 5 % standing envs, heading command on.

| Axis | Final range | First 5000 iterations |
|---|---|---|
| `lin_vel_x` | −1.0 – 2.0 m/s | −0.5 – 1.0 |
| `lin_vel_y` | ±1.0 m/s | ±0.5 |
| `ang_vel_z` | ±1.0 rad/s | ±1.0 |

Curricula: `terrain_levels_vel` (rough only) and `commands_vel` (stage at step 5000·24).

## Rewards

| Term | Weight | Function | Parameters |
|---|---|---|---|
| `track_linear_velocity` | +1.0 | `track_linear_velocity` | std √0.25, z-velocity weight fixed at 2 |
| `track_angular_velocity` | +1.0 | `track_angular_velocity` | std √0.5 |
| `body_orientation_l2` | −1.0 | `body_orientation_l2` | body `base_link` |
| `pose` | +1.0 | `variable_posture` | walking threshold 0.1, running threshold 1.5 |
| `body_ang_vel` | −0.05 | `body_angular_velocity_penalty` | body `base_link` |
| `angular_momentum` | −0.025 | `angular_momentum_penalty` | sensor `robot/root_angmom` |
| `is_terminated` | −200 | mjlab | |
| `joint_acc_l2` | −2.5e-7 | mjlab | |
| `joint_pos_limits` | −10 | mjlab | |
| `action_rate_l2` | −0.05 | mjlab | |
| `foot_gait` | +0.5 | `feet_gait` | period 0.6 s, stance threshold 0.56, trot offsets `[0, 0.5, 0.5, 0]` |
| `foot_clearance` | −1.0 | `feet_clearance` | target height 0.10 m |
| `foot_slip` | −0.25 | `feet_slip` | |
| `soft_landing` | −1e-3 | `soft_landing` | |
| `stand_still` | −1.0 | `stand_still` | |

`pose` stds (Go2 values on Centaur's joints):

| Joint | standing | walking | running |
|---|---|---|---|
| `.*_1` | 0.05 | 0.15 | 0.15 |
| `.*_2` | 0.10 | 0.35 | 0.35 |
| `.*_3` | 0.15 | 0.50 | 0.50 |

### Differences from the trot task's rewards

| Item | Velocity (Unitree) | Trot |
|---|---|---|
| `foot_gait` | stance 0.56 of the cycle, always some foot down | stance 0.5 − flight(v): all four feet in the air between stances |
| `foot_suspension` | – | +0.3 |
| `foot_clearance` target | 0.10 m | 0.15 m |
| `track_linear_velocity` z weight | 2 | 0.5 |
| `pose` stds | Go2's | wider knee (0.3 / 0.8 / 1.0) |
| command | −1 – 2 m/s, standing 5 % | 1 – 2 m/s, no standing |
| SGS | – | `Centaur-Trot-Rough-SGS` |

### Functions (`mdp/rewards.py`)

- `track_linear_velocity`: `exp(-(‖cmd_xy − v_xy‖² + 2·v_z²) / std²)`, body frame.
- `track_angular_velocity`: `exp(-((cmd_yaw − ω_z)² + 0.05·‖ω_xy‖²) / std²)`.
- `body_orientation_l2`: squared xy of projected gravity in the body frame.
- `variable_posture`: `exp(-mean((q − q_default)² / std²))`, std set by command speed.
- `feet_gait`: mean match between foot contact and a period/offset stance schedule, when active.
- `feet_clearance`: `Σ |z_foot − target| · ‖v_foot,xy‖`, when active.
- `feet_slip`, `soft_landing`, `stand_still`, `body_angular_velocity_penalty`, `angular_momentum_penalty`:
  as in [README.md](README.md).
- Defined but unused: `self_collision_cost`, `feet_air_time`, `feet_swing_height`.
