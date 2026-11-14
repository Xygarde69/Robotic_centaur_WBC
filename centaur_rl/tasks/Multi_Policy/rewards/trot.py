"""Trot: diagonal pairs (FR+RL, FL+RR), short or no flight. Spawns at >= 1.20 m/s.

Copied from the trot task (tasks/trot: velocity_env_cfg.py rewards, overridden in
config/centaur/env_cfgs.py), same terms and weights. What changed in the port:

  - command "gait" (lin_x, 0, 0) instead of "twist": no turning, so the roll target of
    body_orientation is upright and track_angular_velocity just holds w_z = 0;
  - the clock is GaitCommand's, not the trot task's fixed 0.6 s period with a
    speed-dependent flight fraction: foot_gait (gait_contact_schedule, exact_weight 0 =
    the trot task's per-foot feet_gait) and foot_suspension both read stance_target, so
    flight phases are wherever TROT.clock schedules them (fixed air time, duty 0.47 ->
    0.39 over the speed range, so flight grows with speed as in the trot task);
  - body_ang_vel uses body-frame roll/pitch rates (the trot task used world-frame x/y;
    the same thing while the base is near upright);
  - pose uses the running stds at every trot command (running_threshold = TROT.v_min,
    as the trot task set it to its minimum command);
  - action_rate_l2 is -0.2 (action scale 0.5 instead of 0.25, as in rewards/walk.py) and
    torque_l2 is added (torque limit 400 Nm here; kept from the walk the trot forks from);
  - foot_slip uses this package's feet_slip (no command gate; the trot is never
    commanded to stand).

Added on top of the port, from the horse data (Robilliard et al. 2007, 187.pdf): touchdown_timing,
excess_contact, speed-normalised foot_clearance and a speed-relative tracking kernel; the
clock itself is in gaits.TROT. See docs/multi_policy.md.

This replaces common_rewards for the trot (table in docs/multi_policy.md).
"""

from __future__ import annotations

import math

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from centaur_rl.tasks.Multi_Policy import mdp
from centaur_rl.tasks.Multi_Policy.gaits import TROT, GaitSpec

from .common import COMMAND, CONTACT_SENSOR, FOOT_SITES

# Trot task values.
TRACK_ANG_STD = math.sqrt(0.5)
TRACK_Z_WEIGHT = 0.5  # flight phases need the base to bounce (walk: 2.0)
TRACK_STD_REL = 0.2  # kernel width = max(spec.tracking_std, 0.2 |v_cmd|): 0.5 -> 0.56 m/s
CLEARANCE_HEIGHT = 0.15  # m
TOUCHDOWN_STD = 0.045  # cycle fraction; horse trot left/right ratios are 0.50 +- 0.03
MOVE_THRESHOLD = 0.1  # m/s, command below this counts as standing

# variable_posture stds (rad), joints .*_1 hip ab/ad, .*_2 hip flex, .*_3 knee.
POSE_STD_STANDING = {r".*_1": 0.05, r".*_2": 0.1, r".*_3": 0.3}
POSE_STD_WALKING = {r".*_1": 0.15, r".*_2": 0.35, r".*_3": 0.8}
POSE_STD_RUNNING = {r".*_1": 0.15, r".*_2": 0.5, r".*_3": 1.0}


def rewards(spec: GaitSpec = TROT) -> dict[str, RewardTermCfg]:
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
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={**cmd, "std": spec.tracking_std, "z_weight": TRACK_Z_WEIGHT, "std_rel": TRACK_STD_REL},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity, weight=1.0, params={**cmd, "std": TRACK_ANG_STD}
    ),
    # Gait clock: fraction of feet matching the schedule (the trot task's feet_gait).
    "foot_gait": RewardTermCfg(
      func=mdp.gait_contact_schedule, weight=0.5, params={**clock, "exact_weight": 0.0}
    ),
    # Landing on time, which for a trot also means diagonal partners landing together:
    # sparse, 4 landings per cycle (~0.2 of its weight per step at full marks at 2.4 Hz).
    "touchdown_timing": RewardTermCfg(
      func=mdp.touchdown_timing, weight=1.5, params={**clock, "std": TOUCHDOWN_STD}
    ),
    # Never 3-4 feet down (a trot has 0-2).
    "excess_contact": RewardTermCfg(func=mdp.feet_excess_contact, weight=-0.3, params=dict(clock)),
    # Suspension: paid while the clock is in a flight phase and all four feet are up.
    # Flight is (1 - 2 duty) of the cycle: 0.06 at 1.2 m/s to 0.21 at 2.8 m/s.
    "foot_suspension": RewardTermCfg(func=mdp.feet_suspension, weight=0.3, params=dict(clock)),
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
        "running_threshold": spec.v_min,
      },
    ),
    # Feet.
    # Divided by the commanded speed (foot speed ~ 2x body speed, so the unnormalised
    # cost grew ~2.3x from 1.2 to 2.8 m/s); weight -1.5 = the trot task's -1.0 at 1.5 m/s.
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.5,
      params={
        **gated,
        "target_height": CLEARANCE_HEIGHT,
        "speed_normalized": True,
        "asset_cfg": feet(),
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip, weight=-0.25, params={"sensor_name": CONTACT_SENSOR, "asset_cfg": feet()}
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing, weight=-1e-3, params={**gated, "sensor_name": CONTACT_SENSOR}
    ),
    # Only active for a standing command (never: the trot starts at 1.2 m/s); kept so the
    # term set matches the trot task.
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-1.0,
      params={**gated, "asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    ),
    # Regularisation.
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    # Action scale 0.5 here (0.25 in the trot task): 4x the trot task's -0.05.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.2),
    "torque_l2": RewardTermCfg(func=mdp.torque_l2, weight=-1e-5),
  }
