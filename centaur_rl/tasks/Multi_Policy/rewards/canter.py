"""Canter: 3-beat asymmetric (RL, then FL + RR, then FR), one short flight after the lead fore.

Starts from the trot's term set (rewards/trot.py) and changes what the canter does
differently, using the horse data in Robilliard, Pfau & Wilson 2007 (187.pdf) and
St. George et al. 2023 (animals-13-01755.pdf). This replaces common_rewards (table in
docs/multi_policy.md). Differences from the trot module:

  - touchdown_timing std 0.05: the horse canter ratios are 0.27 +- 0.03 (fore) and
    +- 0.05 (hind), a bit looser than the trot's 0.50 +- 0.03;
  - diagonal_order (new): the diagonal fore must land before or with its hind. That is
    the only thing separating a canter from a gallop (187 Table 1), and touchdown_timing
    cannot see it because FL and RR are scheduled 0.01 of a cycle apart;
  - excess_contact allows 3 feet down (a canter has 0-3, Table 1), so only 4 is penalised;
  - body pitch is cheap, roll is not: the canter pitches the trunk nose-down through the
    FL + RR diagonal support and back up through the lead-fore support and flight
    (St. George 2023), so pitch is weighted 0.2 against roll 1.0 in body_orientation_l2,
    body_ang_vel; the velocity kernel's z weight is 0.3 (trot 0.5) for the same reason;
  - foot_suspension weight 0.5: one short flight per cycle instead of two, so the term is
    sparser than the trot's;
  - clearance target 0.16 m (the trot's is 0.15) and no stand_still (never commanded).

The clock itself is gaits.CANTER. Weights are first values: nothing has been trained
with them yet.
"""

from __future__ import annotations

import math

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from centaur_rl.tasks.Multi_Policy import mdp
from centaur_rl.tasks.Multi_Policy.gaits import CANTER, GaitSpec

from .common import COMMAND, CONTACT_SENSOR, FOOT_SITES

TRACK_ANG_STD = math.sqrt(0.5)
TRACK_Z_WEIGHT = 0.3  # the base bounces more than in the trot (0.5)
TRACK_STD_REL = 0.2  # kernel width = max(spec.tracking_std, 0.2 |v_cmd|): 0.8 m/s up to 4 m/s
CLEARANCE_HEIGHT = 0.16  # m
TOUCHDOWN_STD = 0.05  # cycle fraction (horse canter ratio s.d. 0.03-0.05)
MOVE_THRESHOLD = 0.1  # m/s, command below this counts as standing
ROLL_WEIGHT = 1.0
PITCH_WEIGHT = 0.2  # pitching is part of the gait

# variable_posture stds (rad), joints .*_1 hip ab/ad, .*_2 hip flex, .*_3 knee.
POSE_STD_STANDING = {r".*_1": 0.05, r".*_2": 0.1, r".*_3": 0.3}
POSE_STD_WALKING = {r".*_1": 0.15, r".*_2": 0.35, r".*_3": 0.8}
POSE_STD_RUNNING = {r".*_1": 0.15, r".*_2": 0.5, r".*_3": 1.0}


def rewards(spec: GaitSpec = CANTER) -> dict[str, RewardTermCfg]:
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
    # Gait clock: fraction of feet matching the schedule.
    "foot_gait": RewardTermCfg(
      func=mdp.gait_contact_schedule, weight=0.5, params={**clock, "exact_weight": 0.0}
    ),
    # Landing on time: sparse, 4 landings per cycle.
    "touchdown_timing": RewardTermCfg(
      func=mdp.touchdown_timing, weight=1.5, params={**clock, "std": TOUCHDOWN_STD}
    ),
    # Canter, not gallop: the diagonal hind (RR) must not land before its fore (FL).
    "diagonal_order": RewardTermCfg(
      func=mdp.diagonal_order, weight=-1.0, params={**clock, "hind": "RR", "fore": "FL"}
    ),
    # Never all four feet down (a canter has 0-3).
    "excess_contact": RewardTermCfg(
      func=mdp.feet_excess_contact, weight=-0.3, params={**clock, "max_feet": 3}
    ),
    # Suspension: paid while the clock is in the flight phase and all four feet are up.
    # Flight is 0.40 - duty of the cycle: 0.02 at 2.24 m/s to 0.08 at 6 m/s.
    "foot_suspension": RewardTermCfg(func=mdp.feet_suspension, weight=0.5, params=dict(clock)),
    # Posture / stability: roll upright, pitch free-ish.
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_roll_pitch,
      weight=-1.0,
      params={"roll_weight": ROLL_WEIGHT, "pitch_weight": PITCH_WEIGHT, "asset_cfg": base()},
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_roll_pitch,
      weight=-0.05,
      params={"roll_weight": ROLL_WEIGHT, "pitch_weight": PITCH_WEIGHT, "asset_cfg": base()},
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
    # Feet. Divided by the commanded speed, as in the trot.
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
    # Regularisation.
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.2),
    "torque_l2": RewardTermCfg(func=mdp.torque_l2, weight=-1e-5),
  }
