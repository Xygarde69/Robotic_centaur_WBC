"""Gallop: right-lead transverse 4-beat (RL, RR, FL, FR), one flight per stride.

Starts from the canter's term set (rewards/canter.py) and changes what the gallop does
differently, using the horse data in Witte, Hirst & Wilson 2006 (4389.pdf: stride timing
of Thoroughbreds at 9-17 m/s) and Self Davies, Spence & Wilson 2019
(12348_Ground-reaction-forces-...pdf: force-plate GRFs at 9-14 m/s). Replaces
common_rewards (table in docs/multi_policy.md). The clock is gaits.GALLOP: per-leg duty
(hind stance longer than fore), one ~70-90 ms flight, at most 2 feet down, 100 Hz control.

Differences from the canter module:

  - footfall_sequence (new) replaces diagonal_order: RL, RR, FL, FR must land in that
    strict order within a stride (transverse gallop, not canter or rotary gallop);
  - unscheduled_flight (new): all feet up outside the clock's flight window is
    penalised. Horses never show a second (mid-stride) flight;
  - flight_duration (new): the flight that ends at RL touchdown should last the clock's
    flight window, not longer (no hopping) and not shorter;
  - grf_peak_excess (new) replaces soft_landing: peak vertical force above 1.3x the
    duty-factor (half-sine) prediction, which force plates confirm in galloping horses;
  - vertical_impulse_balance (new): fore share of the stride's vertical impulse follows
    the measured ~50:50, moving to the hind legs with speed;
  - propulsion_split (new): hind pair pushes, fore pair brakes (net fore-aft impulse);
  - excess_contact max 2 feet (canter 3), touchdown_timing std 0.05 (one 100 Hz step is
    0.031-0.036 of a cycle), pitch weight 0.1 (canter 0.2), z weight 0.2 (canter 0.3),
    foot_suspension 1.0 (the flight is now ~8 control steps, not < 1).

Control rate. The gallop runs at 100 Hz (gaits.GALLOP.decimation = 2) and mjlab pays
every term weight * term * dt. Dense terms are per second, so they keep their weights.
Sparse once-per-event terms (touchdowns, strides) would be worth half as much per event
as at 50 Hz, so their weights are multiplied by EVENT = REFERENCE_DT / control_dt (= 2).
action_rate_l2 is the squared action change per step: for the same smooth motion it is
(dt ratio)^2 smaller per step and paid per second, so its weight is multiplied by EVENT^2
(= 4); -0.4 here is -0.1 at 50 Hz, half the canter's -0.2, because the 60-100 ms stances
need fast joint motion.

Weights are first values: nothing has been trained with them yet.
"""

from __future__ import annotations

import math

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from centaur_rl.tasks.Multi_Policy import mdp
from centaur_rl.tasks.Multi_Policy.gaits import GALLOP, REFERENCE_DT, GaitSpec

from .common import COMMAND, CONTACT_SENSOR, FOOT_SITES

TRACK_ANG_STD = math.sqrt(0.5)
TRACK_Z_WEIGHT = 0.2  # flight time does not grow with speed (Witte 2006): neither should bounce
TRACK_STD_REL = 0.2  # kernel width = max(spec.tracking_std, 0.2 |v_cmd|): 1.0 m/s up to 1.6 m/s
CLEARANCE_HEIGHT = 0.16  # m, as the canter (no swing-height data in either paper)
TOUCHDOWN_STD = 0.05  # cycle fraction, ~1.5 control steps at 100 Hz
MOVE_THRESHOLD = 0.1  # m/s
ROLL_WEIGHT = 1.0
PITCH_WEIGHT = 0.1  # the gallop pitches more than the canter
SEQUENCE = ("RL", "RR", "FL", "FR")  # non-lead hind, lead hind, non-lead fore, lead fore
# Fore share of vertical impulse vs commanded speed (Self Davies 2019 Fig. 3, scaled):
# 0.632 - 0.0195 v = 0.545 at 4.47 m/s, 0.476 at 8 m/s.
FORE_SHARE = (0.632, -0.0195)
FLIGHT_STD = 0.025  # s
GRF_MARGIN = 1.3  # x the duty-factor prediction before the penalty starts

# variable_posture stds (rad), joints .*_1 hip ab/ad, .*_2 hip flex, .*_3 knee (canter's).
POSE_STD_STANDING = {r".*_1": 0.05, r".*_2": 0.1, r".*_3": 0.3}
POSE_STD_WALKING = {r".*_1": 0.15, r".*_2": 0.35, r".*_3": 0.8}
POSE_STD_RUNNING = {r".*_1": 0.15, r".*_2": 0.5, r".*_3": 1.0}


def rewards(spec: GaitSpec = GALLOP) -> dict[str, RewardTermCfg]:
  event = REFERENCE_DT / spec.control_dt  # 2 at 100 Hz, 1 at 50 Hz

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
    # Gait clock: fraction of feet matching the schedule (dense).
    "foot_gait": RewardTermCfg(
      func=mdp.gait_contact_schedule, weight=0.5, params={**clock, "exact_weight": 0.0}
    ),
    # Landing on time: sparse, 4 landings per stride.
    "touchdown_timing": RewardTermCfg(
      func=mdp.touchdown_timing, weight=2.0 * event, params={**clock, "std": TOUCHDOWN_STD}
    ),
    # Transverse gallop order RL < RR < FL < FR within a stride: sparse, once per stride.
    "footfall_sequence": RewardTermCfg(
      func=mdp.footfall_sequence, weight=0.5 * event, params={**clock, "sequence": SEQUENCE}
    ),
    # Never more than 2 feet down (consecutive footfalls overlap, nothing more).
    "excess_contact": RewardTermCfg(
      func=mdp.feet_excess_contact, weight=-0.3, params={**clock, "max_feet": 2}
    ),
    # Flight: paid while the clock is in its flight window and all four feet are up ...
    "foot_suspension": RewardTermCfg(func=mdp.feet_suspension, weight=1.0, params=dict(clock)),
    # ... penalised while all four are up anywhere else (no mid-stride flight, no hops) ...
    "unscheduled_flight": RewardTermCfg(func=mdp.unscheduled_flight, weight=-1.0, params=dict(clock)),
    # ... and scored on its length at the RL touchdown that ends it (sparse).
    "flight_duration": RewardTermCfg(
      func=mdp.flight_duration,
      weight=0.5 * event,
      params={**clock, "landing_foot": SEQUENCE[0], "std": FLIGHT_STD},
    ),
    # Ground reaction forces (Self Davies 2019).
    "grf_peak_excess": RewardTermCfg(
      func=mdp.grf_peak_excess,
      weight=-0.5,
      params={**clock, "fore_share": FORE_SHARE, "margin": GRF_MARGIN},
    ),
    "vertical_impulse_balance": RewardTermCfg(
      func=mdp.vertical_impulse_balance,
      weight=0.3 * event,
      params={**clock, "stride_foot": SEQUENCE[0], "fore_share": FORE_SHARE, "std": 0.05},
    ),
    "propulsion_split": RewardTermCfg(
      func=mdp.propulsion_split, weight=-0.2 * event, params={**clock, "stride_foot": SEQUENCE[0]}
    ),
    # Posture / stability: roll upright, pitch nearly free.
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
    # Feet.
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
    # Regularisation.
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0 * event),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1 * event**2),
    "torque_l2": RewardTermCfg(func=mdp.torque_l2, weight=-1e-5),
  }
