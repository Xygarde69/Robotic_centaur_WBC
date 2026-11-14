# Centaur RL docs

Per-task documentation of the rewards and the functions behind them.

| Doc | Task IDs | Gait / command |
|---|---|---|
| [velocity.md](velocity.md) | `Centaur-Velocity-Rough`, `Centaur-Velocity-Flat` | Unitree's velocity task on the Centaur robot (unmodified rewards), trot gait reward, command up to ±1–2 m/s |
| [trot.md](trot.md) | `Centaur-Trot-Rough`, `Centaur-Trot-Flat`, `Centaur-Trot-Rough-SGS` | flying trot, free (vx, vy, yaw) command, 1–2 m/s |
| [walking.md](walking.md) | `Centaur-Walking-{Rough,Flat}`, `Centaur-Walking-Rough-SGS` | 4-beat walk from `gait.json`, speed on circular arcs + cycle time |
| [pace.md](pace.md) | `Centaur-Pace-{Rough,Flat}`, `Centaur-Pace-Rough-SGS` | pace (same-side legs together) with flight phases |
| [multi_policy.md](multi_policy.md) | `Multi-Gait_straight_flat_SGS` (+ `-<Gait>-<Phase>` stages) | walk → trot → canter → gallop pipeline, one policy per gait, straight line, flat, speed SGS (own layout, see the doc) |

The three tasks share one structure. Everything below applies to all of them unless a task doc says otherwise.
See also [`../REWARDS.md`](../REWARDS.md) for a combined weight/parameter table.

## Layout (per task, `centaur_rl/tasks/<task>/`)

| File | Contents |
|---|---|
| `velocity_env_cfg.py` | `make_velocity_env_cfg()`: base observations, actions, commands, events, rewards, terminations, curriculum |
| `config/centaur/env_cfgs.py` | Centaur overrides, gait constants, rough/flat/SGS/play variants |
| `config/centaur/__init__.py` | `register_mjlab_task(...)` calls (task IDs) |
| `config/centaur/rl_cfg.py` | PPO runner config |
| `mdp/rewards.py` | reward functions |
| `mdp/observations.py` | foot height / air time / contact / forces, gait phase |
| `mdp/terminations.py` | `illegal_contact`, `base_too_low` |
| `mdp/velocity_command.py` | command terms |
| `mdp/curriculums.py` | `terrain_levels_vel`, `commands_vel`, `reward_weight` |
| `mdp/sgs.py` | Success-Guided Sampling curriculum, command and reset event |
| `rl/runner.py` | on-policy runner |

Total reward per step = Σ `weight · term(env)` (positive = reward, negative = penalty).
Rewards tagged "when active" only apply when the command magnitude `‖cmd_xy‖ + |cmd_yaw|`
exceeds `command_threshold` (0.1).

## Robot (`centaur_rl/assets/robots/centaur/centaur_constants.py`)

- Quadruped only (upper body removed from the training MJCF `mjcf/centaur_train.xml`), 12 joints:
  `<leg>_1` hip ab/ad, `<leg>_2` thigh, `<leg>_3` calf, for legs FR, FL, RR, RL.
- Position actuators: kp 150, kd 5, effort limit 120 Nm. Action = joint position target offset, scale 0.25.
- Only the calf-tip spheres (`<leg>_foot_collision`) collide; they act as the feet.
- Default stance: knees bent 30° (one-way, horse-like), thighs back 15°, spawn height 0.64 m.

## Shared environment

- **Sim:** 0.005 s timestep, decimation 4 (0.02 s control), 20 s episodes.
- **Actor observations** (noisy): base angular velocity, projected gravity, command, phase, joint pos/vel (relative),
  last action, height scan (rough only).
- **Critic observations** (clean): actor terms plus base linear velocity, height scan, foot height, foot air time,
  foot contact, foot contact forces.
- **Terminations:** `time_out`; `fell_over` (`bad_orientation`, 70°); `base_too_low` (base less than 0.30 m above
  the mean foot height; replaces Go2's non-foot contact check because only feet collide).
- **Events:** reset base pose/yaw jitter, joint reset, `push_robot` every 5–6 s, foot friction 0.3–1.6,
  encoder bias ±0.015 rad, base COM offset ±5 cm.
- **PPO:** actor/critic MLP (512, 256, 128), ELU, obs normalisation, lr 1e-3 adaptive (desired KL 0.01),
  γ 0.99, λ 0.95, clip 0.2, 24 steps/env, 10001 iterations.

## Reward functions shared by all tasks

Implemented in `mdp/rewards.py` of each task.

| Function | Definition |
|---|---|
| `track_linear_velocity` | `exp(-(‖cmd_xy − v_xy‖² + z_weight·v_z²) / std²)` in the body frame |
| `track_angular_velocity` | `exp(-((cmd_yaw − ω_z)² + 0.05·‖ω_xy‖²) / std²)` |
| `body_angular_velocity_penalty` | squared roll/pitch angular velocity of `base_link` |
| `angular_momentum_penalty` | squared whole-body angular momentum (sensor `robot/root_angmom`) |
| `variable_posture` | `exp(-mean((q − q_default)² / std²))`, std chosen by command speed (standing / walking / running) |
| `feet_clearance` | `Σ |z_foot − target| · ‖v_foot,xy‖`, when active |
| `feet_slip` | `Σ ‖v_foot,xy‖²` over feet in contact, when active |
| `soft_landing` | summed contact-force magnitude at first contact, when active |
| `stand_still` | squared joint deviation from default, only when the command is below threshold |
| mjlab built-ins | `is_terminated`, `joint_acc_l2`, `joint_pos_limits`, `action_rate_l2` |

Defined but not wired into any config: `self_collision_cost`, `feet_air_time`, `feet_swing_height`,
`feet_gait_schedule`.

## Success-Guided Sampling (SGS)

Implemented in `mdp/sgs.py` (arXiv:2610.12465). Only the `*-Rough-SGS` tasks exist; the flat tasks use the plain curricula. SGS only changes **what is sampled**, not the rewards. It replaces the
terrain-level and command curricula.

- `SuccessGuidedSampling` (curriculum term): keeps the last 100 outcomes per configuration, scores each with a Beta
  kernel (target 0.66, κ 5), and at every reset samples the next configuration from `softmax(log w / T)` (T = 2).
  Moves the env to its terrain tile.
- `reset_root_state_sgs` (reset event): places the robot at the episode's spawn pose.
- `SgsVelocityCommand`: holds the command fixed for the episode and accumulates the tracking error for the
  success test.
- **Success** of an episode: mean velocity (not absolute error) matches the command within tolerance, ignoring the
  first second after reset; falling on the tile is failure. On tiled terrain the outcome is frozen once the robot
  leaves its tile.
- Logged under `Curriculum/sgs/`: `episode_success`, `episode_left_tile_frac`, `success_easy_levels`,
  `success_hard_levels`, `sampled_<axis>`, `sampled_terrain_level`, `configs_seen_frac`, `configs_frontier_frac`.
