# Pace task

Task IDs: `Centaur-Pace-Rough`, `-Flat`, `-Rough-SGS`, `-Flat-SGS`.
Code: `centaur_rl/tasks/pace/`. Shared structure, robot and reward functions: see [README.md](README.md).

A copy of the [walking task](walking.md) (same arc command, turn lean, gait clock, observations) with a different
gait: a **pace with flight phases**, where the two legs on one side move together.

## Gait

Cycle time fixed at 0.6 s (the trot's); the command still carries it as its 4th entry. A leg is in stance while
`(cycles + offset) % 1 < 0.5 − flight(v)`, with offsets `[FR 0.5, FL 0, RR 0.5, RL 0]`.

One cycle: right pair (FR+RR) in the air, left pair on the ground → flight (all four in the air) → left pair
(FL+RL) in the air, right pair on the ground → flight.

flight(v) = 0.02 + 0.06·v of the cycle (clamped at 2 m/s):

| v | one flight phase | foot air time |
|---|---|---|
| 0.5 m/s | 0.030 s | 0.330 s |
| 1.0 m/s | 0.048 s | 0.348 s |
| 1.69 m/s | 0.073 s | 0.373 s |

No startup hold from rest: the right pair lifts at cycle 0, the left pair stands first.

## Command

Same as walking (`WalkVelocityCommand`): `lin_x` 0 – 1.687 m/s, `lin_y` 0, `ang_z = lin_x·κ` on arcs
(R ≥ 1 m, lateral acceleration ≤ 0.2 g, 25 % straight), 5 % standing. Cycle-time range collapsed to (0.6, 0.6).

## Rewards

Identical to walking except the following.

| Term | Weight | Function | Notes |
|---|---|---|---|
| `foot_gait` | +1.5 | `feet_gait_clock` | on the pace schedule, `exact_weight` 0.5 |
| `foot_contact_in_swing` | −0.5 | `feet_contact_in_swing` | on the pace schedule |
| `foot_suspension` | +0.3 | `feet_suspension` | all feet off the ground during a flight phase |
| `track_linear_velocity` | +1.0 | | z weight 0.5 (flight needs base bounce) |

The full set otherwise matches [walking.md](walking.md): `track_angular_velocity` (+1.0), `body_orientation_l2`
(−1.0, turn-lean aware), `pose` (+1.0, hip std 0.4), `body_ang_vel` (−0.05), `angular_momentum` (−0.025),
`is_terminated` (−200), `joint_acc_l2`, `joint_pos_limits` (−10), `action_rate_l2` (−0.05),
`foot_clearance` (−0.3, 0.13 m), `foot_slip` (−0.25), `soft_landing`, `stand_still`.

### Task-specific functions (`mdp/rewards.py`)

All of walking's, plus:

- `gait_flight_fraction`: speed → flight-fraction map.
- `feet_suspension` (`_WalkClockTerm`): 1 while the clock is in a flight phase and no foot is in contact. Averages
  2 × flight fraction per step when achieved (0.16 at 1 m/s, 0.24 at top speed), i.e. at most +0.05 / +0.07 at weight 0.3.
- `feet_gait_clock` / `feet_contact_in_swing`: use the stance threshold `0.5 − flight(v)`.
  The all-four-match half matters because a trot matches half of the pace's foot states at any time but never all four.

## SGS variants

Axis-bin design (`mdp/sgs.py`): a configuration is terrain level × terrain type × one cell of the command grid.

| | Rough-SGS | Flat-SGS |
|---|---|---|
| axes | `lin_vel_x` [0.3, 1.69], `turn` [−1, 1] | same |
| bins | 6 × 5 × 200 tiles = 6,000 configs | 24 × 15 = 360 configs |

`turn` sets curvature = turn · κ_max(v), yaw = v · curvature (arcs only); the middle bin is exactly straight
(`snap_center`). 5 % of episodes are standing and sit outside the config set. Success tolerance:
speed max(0.15, 0.25·v) m/s, yaw rate 0.2 rad/s; the first second is ignored.
