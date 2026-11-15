"""Reward terms shared by every gait (PLACEHOLDER weights).

Each gait module starts from `common_rewards(spec)` and adds / overrides its own terms.
Weights here are deliberately plain; tune them per gait in rewards/<gait>.py.
"""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from centaur_rl.tasks.Multi_Policy import mdp
from centaur_rl.tasks.Multi_Policy.gaits import GaitSpec

COMMAND = "gait"
CONTACT_SENSOR = "feet_ground_contact"
FOOT_SITES = ("FR", "FL", "RR", "RL")

# Sharp yaw-rate term added on top of every gait's track_angular_velocity. The main term's
# kernel (std 0.5 / 0.71 rad/s) barely separates turning from not turning: ignoring a
# 0.2 rad/s yaw command still scores 0.85-0.92 of it. This one scores 0.53 there (std
# 0.25) and ignores roll/pitch rate (xy_weight 0). Straight, w_cmd = 0: it holds heading.
TRACK_YAW_FINE_STD = 0.25  # rad/s
TRACK_YAW_FINE_WEIGHT = 0.5


# Lean into turns: positive kernel on the turn's equilibrium lean, every gait. See
# mdp.track_turn_lean for why the L2 orientation term does not produce the lean.
TRACK_TURN_LEAN_STD = 0.07  # in g_y (= sin of the roll error), ~4 deg
TRACK_TURN_LEAN_WEIGHT = 0.5


def track_turn_lean() -> RewardTermCfg:
  return RewardTermCfg(
    func=mdp.track_turn_lean,
    weight=TRACK_TURN_LEAN_WEIGHT,
    params={
      "command_name": COMMAND,
      "std": TRACK_TURN_LEAN_STD,
      "gate_by_progress": True,
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
    },
  )


# Alive bonus for the forked gaits (trot, canter, gallop). A forked actor starts bad:
# in gaits_v4 the trot fork netted about -4.1 per second alive (action_rate, foot_slip and
# foot_clearance against +1.75 of positive terms) while is_terminated costs only
# -200 * 0.02 = -4 once, so PPO learned to drop the base within ~0.2 s. + weight per second
# (per-step reward is weight * dt, so this holds at 100 Hz too) makes surviving worth more
# than dying from the first iterations. Tried 5.0, 1.0 and 0.1; now 0.75 (+15 over a full
# 20 s episode). The yaw / lean terms are gated by forward progress, so standing still
# earns this bonus but not them.
ALIVE_WEIGHT = 0.75


# Regularisation terms scaled in by mdp.PenaltyRamp when a gait's budget sets
# penalty_ramp_iterations (stages trained from scratch).
RAMPED_PENALTIES = ("action_rate_l2", "joint_acc_l2", "foot_slip", "torque_l2")


def alive_bonus() -> RewardTermCfg:
  return RewardTermCfg(func=mdp.is_alive, weight=ALIVE_WEIGHT)


def track_yaw_fine() -> RewardTermCfg:
  return RewardTermCfg(
    func=mdp.track_angular_velocity,
    weight=TRACK_YAW_FINE_WEIGHT,
    params={"command_name": COMMAND, "std": TRACK_YAW_FINE_STD, "xy_weight": 0.0, "gate_by_progress": True},
  )


def common_rewards(spec: GaitSpec) -> dict[str, RewardTermCfg]:
  feet = SceneEntityCfg("robot", site_names=FOOT_SITES)
  return {
    # Task.
    "track_forward_velocity": RewardTermCfg(
      func=mdp.track_forward_velocity,
      weight=1.5,
      params={"command_name": COMMAND, "std": spec.tracking_std},
    ),
    "straight_line": RewardTermCfg(func=mdp.straight_line, weight=0.5, params={"std": 0.5}),
    # Clock (generic; every gait module re-weights or replaces it).
    "gait_schedule": RewardTermCfg(
      func=mdp.gait_contact_schedule,
      weight=1.0,
      params={"command_name": COMMAND, "sensor_name": CONTACT_SENSOR, "exact_weight": 0.5},
    ),
    # Posture / stability.
    "base_tilt": RewardTermCfg(func=mdp.base_roll_pitch_l2, weight=-1.0),
    "base_tilt_rate": RewardTermCfg(func=mdp.base_roll_pitch_rate_l2, weight=-0.05),
    "base_vertical_vel": RewardTermCfg(func=mdp.base_vertical_velocity_l2, weight=-0.5),
    # Regularisation.
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "torque_l2": RewardTermCfg(func=mdp.torque_l2, weight=-1e-5),
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip, weight=-0.1, params={"sensor_name": CONTACT_SENSOR, "asset_cfg": feet}
    ),
  }
