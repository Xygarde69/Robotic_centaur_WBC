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
