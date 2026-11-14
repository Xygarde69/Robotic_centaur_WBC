# Walking task (clocked walk)

Task IDs: `Centaur-Walking-Rough`, `-Flat`, `-Rough-SGS`, `-Flat-SGS`.
Code: `centaur_rl/tasks/walking/`. Shared structure, robot and reward functions: see [README.md](README.md).

A 4-beat walk whose contact schedule comes from `mdp/gait.json`. The gait file fixes the **shape** of the cycle
(duty factor 0.524, leg start order FR → RL → FL → RR); the cycle **length** is a command entry.

## Command (`WalkVelocityCommand`)

The command is `(lin_x, 0, ang_z = lin_x · κ, cycle_time)`: the robot walks circular arcs and never turns on the spot.

| Item | Value |
|---|---|
| `lin_vel_x` | 0 – min(2, v_max) = 0 – 1.687 m/s (first 5000 iterations: half of that) |
| `v_max` | `sqrt(Fr·g·L)`, Fr 0.5, L 0.58 m |
| curvature κ | uniform in ±min(1/R_min, a_lat/v²), R_min 1 m, a_lat 0.2 g; 25 % of commands straight |
| cycle time | [0.509, 1.66] s, feasible for the speed and turn (faster → shorter); stride kept in [0.15, 0.45] m |
| standing | 5 % of envs |

Helpers in `mdp/velocity_command.py`: `walk_max_curvature`, `sample_walk_curvature`, `walk_cycle_time_range`.
The command term also runs the **gait clock**: cycles since walking started, zero while standing, restarted at
every reset and standing → walking switch. The viewer joystick is clamped to the same arcs.

## Gait

`load_gait_schedule` reads `gait.json`; `gait_leg_clocks` converts it into per-leg offsets, stance fraction and
lift-off phases. From rest each leg stays down until its first lift-off (FR 0, RL 0.25, FL 0.5, RR 0.75 of a
cycle), so the walk begins with FR's long stride as in `gait.json` P0; afterwards P8 ≡ P0.

## Rewards

| Term | Weight | Function | Notes |
|---|---|---|---|
| `track_linear_velocity` | +1.0 | `track_linear_velocity` | std 0.5 |
| `track_angular_velocity` | +1.0 | `track_angular_velocity` | std 0.5 |
| `body_orientation_l2` | −1.0 | `body_orientation_roll_pitch` | roll measured from the turn lean, pitch from upright |
| `pose` | +1.0 | `variable_posture` | hip ab/ad std 0.4 when moving (two-foot supports need CoM shift) |
| `body_ang_vel` | −0.05 | `body_angular_velocity_roll_pitch` | body-frame roll rate² + pitch rate² |
| `angular_momentum` | −0.025 | `angular_momentum_penalty` | |
| `is_terminated` | −200 | mjlab | |
| `joint_acc_l2` | −2.5e-7 | mjlab | |
| `joint_pos_limits` | −10 | mjlab | |
| `action_rate_l2` | −0.05 | mjlab | |
| `foot_gait` | +1.5 | `feet_gait_clock` | 0.5 · per-foot match + 0.5 · [all four match] on the walk clock |
| `foot_contact_in_swing` | −0.5 | `feet_contact_in_swing` | fraction of feet on the ground while their clock says swing |
| `foot_clearance` | −0.3 | `feet_clearance` | target 0.13 m |
| `foot_slip` | −0.25 | `feet_slip` | |
| `soft_landing` | −1e-3 | `soft_landing` | |
| `stand_still` | −1.0 | `stand_still` | |

The all-four-match half of `foot_gait` exists because a best-aligned trot matches 75 % of the walk's foot states
per foot but all four only 45 % of the time; a perfect walk earns 1.5 per step vs ≈0.9 for a trot.

### Task-specific functions (`mdp/rewards.py`)

- `body_orientation_roll_pitch`: `(g_y − sin θ)² + pitch²`, with `θ = atan(v²κ / g)` the equilibrium lean for the
  measured forward speed and commanded curvature (θ = 0 when straight, ≤ ~11°).
- `body_angular_velocity_roll_pitch`: weighted body-frame roll and pitch rate squares.
- `walk_gait_cycles(env, command_name, lookahead)`: cycle count and walking mask from the command's clock.
- `_WalkClockTerm`: base class for clock-based rewards (resolves foot order, offsets, lift phases).
- `feet_gait_clock`: per-foot + all-four contact match against the clock; zero while standing.
- `feet_contact_in_swing`: penalty for ground contact in the scheduled swing.
- `load_gait_schedule`, `gait_timing`, `gait_leg_clocks`: gait JSON → clock parameters.
- `gait_step_table`, `resolve_gait_step_table`, `gait_stance_target`, `feet_gait` / `feet_gait_schedule`:
  table-based schedule used by the `gait_target` observation (one-step lookahead).
- `variable_posture`: pose reward as in the velocity task.

### Observations

`walk_phase`: `[sin, cos, min(cycles, 1)]` of the walk clock, zero when standing. The third entry separates
the startup from rest from the repeating cycle. `gait_target` gives the scheduled stance per foot for the next step.

## Differences from the velocity task

| Item | Velocity | Walking |
|---|---|---|
| gait | flying trot | walk (`gait.json`) |
| `foot_gait` | +0.5, `feet_gait` | +1.5, `feet_gait_clock` |
| `foot_contact_in_swing` | – | −0.5 |
| `foot_clearance` | −1.0, 0.15 m | −0.3, 0.13 m |
| `foot_suspension` | +0.3 | – |
| `track_linear_velocity` z weight | 0.5 | default (2) |
| roll/pitch terms | world x/y | body-frame, turn-lean aware |
| `pushes` | ±0.5 m/s | scaled by min(1, 2·v_max) (no-op) |

## SGS variants

Command per config = `(lin_x, 0, lin_x·κ, cycle time)`; the turn sharpness is a per-config parameter.
Success tolerance: `max_err_xy = min(0.3, 0.4·v_max)` m/s, `max_err_yaw = 0.2` rad/s, 5 % standing episodes.

| | Rough-SGS | Flat-SGS |
|---|---|---|
| configs | 10 levels × 20 tiles × 4 spawns × 32 commands = 25,600 | 4 spawns × 1024 commands = 4,096 |

The walking SGS still uses the older random-command design (spawn + command lists), not the axis-bin design of
the velocity and pace tasks.
