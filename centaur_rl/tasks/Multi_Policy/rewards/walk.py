"""Walk: 4-beat lateral sequence, always >= 2 feet down, no flight. Starts from rest.

Copied from the walking task (tasks/walking: velocity_env_cfg.py rewards, overridden in
config/centaur/env_cfgs.py), same terms and weights. What changed in the port:

  - command "gait" (lin_x, 0, 0) instead of "twist": no turning, so the roll target of
    body_orientation is upright (no turn lean) and track_angular_velocity just holds
    w_z = 0;
  - the gait clock is GaitCommand's (stride frequency from the speed, not a commanded
    cycle time). The walking task's startup hold (each foot down until its first
    lift-off when starting from rest) lives in GaitCommand.stance_target;
  - tracking_std is the gait spec's 0.35 (the walking task used 0.5), action_rate_l2 is
    -0.2 (action scale 0.5), torque_l2 is added, and touchdown_timing is new (on-time
    landings; see docs/multi_policy.md);
  - foot_slip uses this package's feet_slip (identical maths without the command gate;
    the walk is never commanded to stand).

This replaces common_rewards for the walk (table in docs/multi_policy.md).
"""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from centaur_rl.tasks.Multi_Policy import mdp
from centaur_rl.tasks.Multi_Policy.gaits import WALK, GaitSpec

from .common import COMMAND, CONTACT_SENSOR, FOOT_SITES

# Walking task values (except where noted in rewards()).
TRACK_ANG_STD = 0.5
CLEARANCE_HEIGHT = 0.13  # m
MOVE_THRESHOLD = 0.1  # m/s, command below this counts as standing

# variable_posture stds (rad), joints .*_1 hip ab/ad, .*_2 hip flex, .*_3 knee.
POSE_STD_STANDING = {r".*_1": 0.05, r".*_2": 0.1, r".*_3": 0.3}
POSE_STD_WALKING = {r".*_1": 0.4, r".*_2": 0.35, r".*_3": 0.8}
POSE_STD_RUNNING = {r".*_1": 0.4, r".*_2": 0.5, r".*_3": 1.0}


def rewards(spec: GaitSpec = WALK) -> dict[str, RewardTermCfg]:
  # Fresh SceneEntityCfg per term: the manager resolves each one in place.
  def base() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=("base_link",))

  def feet() -> SceneEntityCfg:
    return SceneEntityCfg("robot", site_names=FOOT_SITES)

  cmd = {"command_name": COMMAND}
  gated = {**cmd, "command_threshold": MOVE_THRESHOLD}
  clock = {**cmd, "sensor_name": CONTACT_SENSOR}
  return {
    # Task.
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity, weight=1.0, params={**cmd, "std": spec.tracking_std}
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity, weight=1.0, params={**cmd, "std": TRACK_ANG_STD}
    ),
    # Gait clock: 0.5 * per-foot match + 0.5 * [all four match]. The all-match half is
    # what separates a walk from a best-aligned trot (75% per-foot, 45% all-four).
    "foot_gait": RewardTermCfg(
      func=mdp.gait_contact_schedule, weight=1.5, params={**clock, "exact_weight": 0.5}
    ),
    "foot_contact_in_swing": RewardTermCfg(
      func=mdp.feet_contact_in_swing, weight=-0.5, params=dict(clock)
    ),
    # Landing on time: sparse, 4 landings per cycle, so ~0.1 of its weight per step at
    # full marks (the contact match above cannot tell a late landing from a wrong foot).
    "touchdown_timing": RewardTermCfg(
      func=mdp.touchdown_timing, weight=3.0, params={**clock, "std": 0.08}
    ),
    # Posture / stability.
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_roll_pitch,
      weight=-1.0,
      params={"roll_weight": 1.0, "pitch_weight": 1.0, "asset_cfg": base()},
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_roll_pitch,
      weight=-0.05,
      params={"roll_weight": 1.0, "pitch_weight": 1.0, "asset_cfg": base()},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty, weight=-0.025, params={"sensor_name": "robot/root_angmom"}
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        **cmd,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "std_standing": POSE_STD_STANDING,
        "std_walking": POSE_STD_WALKING,
        "std_running": POSE_STD_RUNNING,
        "walking_threshold": MOVE_THRESHOLD,
        "running_threshold": 1.5,
      },
    ),
    # Feet.
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-0.3,
      params={**gated, "target_height": CLEARANCE_HEIGHT, "asset_cfg": feet()},
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip, weight=-0.25, params={"sensor_name": CONTACT_SENSOR, "asset_cfg": feet()}
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing, weight=-1e-3, params={**gated, "sensor_name": CONTACT_SENSOR}
    ),
    # Only active for a standing command (none in the walk's SGS range, which starts at
    # 0.15 m/s); kept so the term set matches the walking task.
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-1.0,
      params={**gated, "asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    ),
    # Regularisation.
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    # Action scale is 0.5 here (0.25 in the walking task), so the same action change is
    # twice the joint motion: 4x the penalty keeps the smoothness pressure per radian.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.2),
    # The walking task had none; the torque limit is 400 Nm here instead of 120.
    "torque_l2": RewardTermCfg(func=mdp.torque_l2, weight=-1e-5),
  }
