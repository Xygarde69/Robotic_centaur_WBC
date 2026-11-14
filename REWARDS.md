# Reward functions

Active reward terms for the Centaur trot task (`Centaur-Trot-*`; the unmodified Unitree velocity task is `Centaur-Velocity-*`, see `docs/velocity.md`). Defaults live in
`centaur_rl/tasks/trot/velocity_env_cfg.py`; Centaur-specific overrides are in
`centaur_rl/tasks/trot/config/centaur/env_cfgs.py`; implementations are in
`centaur_rl/tasks/trot/mdp/rewards.py` (or mjlab built-ins where noted).

Total reward per step = sum over terms of `weight * term(env)`. Positive weight = reward,
negative weight = penalty. "Active" means the command magnitude
`‖cmd_xy‖ + |cmd_yaw|` exceeds `command_threshold` (0.1), i.e. the robot is asked to move.

## Active terms

| Term | Weight | Function | Key params |
|---|---|---|---|
| `track_linear_velocity` | +1.0 | `mdp.track_linear_velocity` | std = √0.25, z weight 0.5 |
| `track_angular_velocity` | +1.0 | `mdp.track_angular_velocity` | std = √0.5 |
| `body_orientation_l2` | −1.0 | `mdp.body_orientation_l2` | body = `base_link` |
| `pose` | +1.0 | `mdp.variable_posture` | per-joint stds below; walking thr. 0.1, running thr. 1.0 |
| `body_ang_vel` | −0.05 | `mdp.body_angular_velocity_penalty` | body = `base_link` |
| `angular_momentum` | −0.025 | `mdp.angular_momentum_penalty` | sensor `robot/root_angmom` |
| `is_terminated` | −200.0 | mjlab `is_terminated` | one-shot penalty on failure termination |
| `joint_acc_l2` | −2.5e-7 | mjlab `joint_acc_l2` | |
| `joint_pos_limits` | −10.0 | mjlab `joint_pos_limits` | |
| `action_rate_l2` | −0.05 | mjlab `action_rate_l2` | |
| `foot_gait` | +0.5 | `mdp.feet_gait` | period 0.6 s, stance fraction 0.5 − flight(v), trot offset `[0, 0.5, 0.5, 0]` |
| `foot_suspension` | +0.3 | `mdp.feet_suspension` | same schedule; all feet off the ground during a flight phase |
| `foot_clearance` | −1.0 | `mdp.feet_clearance` | target height 0.15 m |
| `foot_slip` | −0.25 | `mdp.feet_slip` | sensor `feet_ground_contact` |
| `soft_landing` | −1e-3 | `mdp.soft_landing` | sensor `feet_ground_contact` |
| `stand_still` | −1.0 | `mdp.stand_still` | joints `.*` |

## Term descriptions

### Task (velocity tracking)
- **`track_linear_velocity`** — `exp(-(‖cmd_xy − v_xy‖² + z_weight·v_z²) / std²)`. Rewards matching the commanded
  body-frame xy velocity and penalizes vertical velocity. Range (0, 1]. `z_weight` is 0.5 (Go2: 2): the
  flight phases need the base to bounce.
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
  Flying trot: a leg is in stance while its phase < 0.5 − flight(v), so after each diagonal pair's
  stance all four feet are scheduled in the air (there is no all-feet-down transition).
  flight(v) = 0.02 + 0.06·‖cmd_xy‖ of the cycle (clamped at 2 m/s), per flight phase:

  | ‖cmd_xy‖ | one flight phase | foot air time | stance |
  |---|---|---|---|
  | 1.0 m/s | 0.048 s | 0.348 s | 0.252 s |
  | 1.5 m/s | 0.066 s | 0.366 s | 0.234 s |
  | 2.0 m/s | 0.084 s | 0.384 s | 0.216 s |
- **`foot_suspension`** — 1 while the schedule is in a flight phase and no foot touches the ground.
  Summed over a cycle it is the flight time achieved, so faster commands (longer flight) earn more:
  on average up to 0.16 (1 m/s) to 0.28 (2 m/s) per step before the weight.
- **`foot_clearance`** — `Σ |z_foot − 0.15| · ‖v_foot,xy‖`; penalizes feet moving fast at the wrong
  height, encouraging a consistent swing height. Only when active.
- **`foot_slip`** — `Σ ‖v_foot,xy‖²` over feet in contact. Only when active.
- **`soft_landing`** — sum of contact-force magnitudes at the moment of first foot contact. Only when active.

### Regularization / safety
- **`joint_acc_l2`** — squared joint accelerations (smoothness).
- **`joint_pos_limits`** — penalty for joints beyond their soft position limits.
- **`action_rate_l2`** — squared change in action between steps.
- **`is_terminated`** — fires on failure terminations (not time-outs).

### Commands
`lin_vel_x` ∈ [1, 2] m/s always (first 5000 iterations: [1, 1.5]), no standing envs, no backward.
`lin_vel_y` ∈ ±0.5 (first stage ±0.3), `ang_vel_z` ∈ ±1. SGS variants use the same ranges, no standing configs.

## Defined but not used
These exist in `mdp/rewards.py` but are not in the reward config:
- `self_collision_cost` — counts contact-force hits above a threshold.
- `feet_air_time` — rewards single-stance air time near a target.
- `feet_swing_height` — class; penalizes peak swing height error at landing.

## SGS tasks
The `*-SGS` tasks use the same reward set. SGS only changes what is sampled, not the rewards.
Implementation: `mdp/sgs.py` of each task (trot and pace share the design below; the
walking task still has the older random-command version).

A configuration is terrain level × terrain type × one cell of a command-parameter grid. Each
episode draws its command uniformly inside its cell; spawn offset and heading are random per
reset and not part of the configuration.

| | `Centaur-Trot-Rough-SGS` | `Centaur-Pace-Rough-SGS` |
|---|---|---|
| command axes | `lin_vel_x` [1, 2], `lin_vel_y` ±0.5, `ang_vel_z` ±1 | `lin_vel_x` [0.3, 1.69], `turn` [−1, 1] |
| meaning of `turn` | – | curvature = turn · κ_max(v), yaw = v · curvature (arcs only); middle bin exactly straight |
| bins | 4 × 3 × 3, × 200 tiles = 7,200 configs | 6 × 5, × 200 tiles = 6,000 configs |
| standing | none | 5% of episodes, outside the config set (not scored) |
| speed tolerance | max(0.3, 0.2·‖cmd‖) m/s | max(0.15, 0.25·v) m/s |
| yaw-rate tolerance | 0.3 rad/s | 0.2 rad/s |

Success of an episode (sampling signal, not a reward):
- It is decided on the configuration's own 8 m tile. Once the robot is more than 3.75 m from the
  tile origin in x or y, the outcome is frozen: success if it tracked the command while on the
  tile. Falling on the tile is a failure. A turning command that stays on the tile is judged at
  time-out.
- "Tracked" compares the mean velocity (not the mean absolute error, so the ripple inside each
  stride does not count) with the command, leaving out the first second after reset.
- Logged under `Curriculum/sgs/`: `episode_success`, `episode_left_tile_frac`,
  `success_easy_levels`, `success_hard_levels`, `sampled_<axis>`, `sampled_terrain_level`,
  `configs_seen_frac`, `configs_frontier_frac`.

## Walking tasks (`Centaur-Walking-*`)
`centaur_rl/tasks/walking/` starts from the reward set above and is tuned so the contact
schedule in `centaur_rl/tasks/walking/mdp/gait.json` is the dominant objective. Everything below
is derived from the gait file in `config/centaur/env_cfgs.py`, so swapping the JSON rescales it.

Walking along circular arcs: the command is `lin_x` in (0, v_max), `lin_y = 0`,
`ang_z = lin_x · κ` with a turn curvature κ = 1/R, plus a gait cycle time as a 4th command entry
(observed by the policy). A yaw rate therefore always comes with forward motion: the robot
walks a circle of radius R, it never turns on the spot.

| Item | Trot task | Walking task (walk, gait.json shape, duty 0.524) |
|---|---|---|
| `foot_gait` | +0.5, `feet_gait` (phase offsets) | +1.5, `feet_gait_clock`: 0.5 · per-foot match + 0.5 · [all 4 match], walk clock, startup from rest |
| `foot_contact_in_swing` | – | −0.5, fraction of feet on the ground while their clock says swing |
| `foot_clearance` | −1.0, target 0.15 m | −0.3, target 0.13 m |
| `body_orientation_l2` | −1.0, roll² + pitch² | −1.0, (g_y − sin θ)² + pitch², θ = atan(v²κ / g) turn lean (`body_orientation_roll_pitch`) |
| `body_ang_vel` | −0.05, world x/y rates | −0.05, body-frame roll rate² + pitch rate² |
| command `lin_vel_x` | (1, 2) | (0, min(2, v_max)) = (0, 1.687) |
| command `lin_vel_y` | (−0.5, 0.5) | 0 |
| command `ang_vel_z` | (−1, 1), independent | `lin_x · κ`, κ ∈ ±min(1/1 m, 0.2 g / v²), 25% straight (κ = 0) |
| command cycle time | – | [0.509, 1.66] s, feasible range for the speed and turn (faster → shorter) |
| `track_linear_velocity` std | 0.5 | 0.5 |
| `track_angular_velocity` std | 0.707 | 0.5 |
| `pose` hip ab/ad std (walking, running) | 0.15 | 0.4 |
| `push_robot` velocities | ±0.5 m/s etc. | same (× min(1, 2·v_max)) |
| SGS success thresholds (xy, yaw) | max(0.3, 0.2·‖cmd‖), 0.3 | 0.3, 0.2 |
| SGS config | tile × (lin_x, lin_y, yaw) cell | command = (lin_x, 0, lin_x · κ, cycle time); κ (turn sharpness) per config |
| `phase` observation | sin/cos, 0.6 s period | `walk_phase`: sin/cos of the walk clock + first-cycle progress |

- **Max walk speed** `v_max = sqrt(Fr · g · L)` with Froude number 0.5 (walk–trot transition)
  and leg length L = 0.58 m → 1.687 m/s.
- **Cycle time.** A planted foot sweeps `v · duty · T` under its hip; that stride is kept in
  [0.15, 0.45] m, so `T ∈ [0.15 / (duty·v), 0.45 / (duty·v)]` clipped to [0.509, 1.66] s.
  0.509 s is the cycle that reaches v_max with a full 0.45 m stride (0.24 s swing); 1.66 s is
  gait.json's own cycle. The 1.66 s cycle is allowed from 0.17 to 0.52 m/s and forced below 0.17 m/s, where strides drop under 0.15 m. All gait.json timings
  scale with T (shape and duty factor are fixed).
- **Turns** (`sample_walk_curvature`): κ uniform in [−κ_max(v), κ_max(v)] with
  κ_max = min(1 / R_min, a_lat / v²), R_min = 1 m, a_lat = 0.2 g (~11° lean). So the 1 m
  circle is allowed up to 1.40 m/s (yaw up to 1.4 rad/s); at v_max R ≥ 1.45 m. 25% of
  walking commands are straight. κ = 0 whenever lin_x = 0. On a turn the outer feet
  (0.23 m off the midline) move at `v · (1 + 0.23 |κ|)`, and that speed sets the cycle-time
  range. In SGS the curvature is a fixed per-config parameter (rough-SGS now uses 32 commands
  per tile/spawn, 25,600 configs). The viewer joystick is clamped to the same arcs.
- **Turn lean** (`body_orientation_roll_pitch`): walking a circle of radius r = 1/κ at speed v
  needs a centripetal acceleration v²/r, balanced by leaning into the turn by
  θ = atan(v² / (r g)). The roll penalty is measured from that lean: `(g_y − sin θ)²`, with
  g_y the body-frame gravity y component, v the measured forward speed at that instant and κ
  the commanded curvature (θ = 0 when straight; at most ~11° with the 0.2 g cap).
- **Gait clock** (`WalkVelocityCommand`): counts cycles since walking started, zero while
  standing, restarted at every episode reset and standing → walking switch.
- **`foot_gait`** (`feet_gait_clock`): as the trot, the fraction of feet whose contact matches
  their clock; stance while `(cycles + offset) % 1 < duty`. From rest each leg stays down
  until its first lift-off (FR 0, RL 0.25, FL 0.5, RR 0.75 of a cycle), so the walk starts with
  FR's long stride exactly as gait.json P0; afterwards P8 ≡ P0. Zero while standing.
- Offsets and lift phases are given in `LEG_NAMES` order and permuted to the contact sensor's
  foot order (model order FL, FR, RL, RR) by `feet_gait_clock`.

## Pace tasks (`Centaur-Pace-*`)
`centaur_rl/tasks/pace/` is a copy of the walking task (same command term, arcs, turn lean,
clock, observations, SGS). Only the gait and what it needs differ:

| Item | Walking task | Pace task |
|---|---|---|
| gait | gait.json walk, duty 0.524 | pace: FR+RR in the air → flight → FL+RL in the air → flight |
| cycle time | [0.509, 1.66] s by speed | fixed 0.6 s (the trot's); still the 4th command entry |
| stance fraction | 0.524 | 0.5 − flight(v), flight(v) = 0.02 + 0.06·v of the cycle (as the flying trot) |
| startup from rest | per-leg lift phases | none needed: the right pair lifts at cycle 0, the left pair stands first |
| `foot_gait` | +1.5, `feet_gait_clock`, exact weight 0.5 | same, on the pace schedule |
| `foot_contact_in_swing` | −0.5 | same, on the pace schedule |
| `foot_suspension` | – | +0.3, `feet_suspension`: all feet off the ground during a flight phase |
| `track_linear_velocity` z weight | 2 | 0.5 |

One flight phase lasts 0.030 s at 0.5 m/s, 0.048 s at 1 m/s and 0.073 s at 1.69 m/s (foot air
time 0.330 / 0.348 / 0.373 s). Speed range, turns and everything else are the walking task's.
