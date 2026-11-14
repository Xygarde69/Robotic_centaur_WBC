"""Centaur velocity environment configurations.

Same paradigm as the Unitree Go2 velocity task (unitree_rl_mjlab): 12 leg joints are the
action/observation space. The training MJCF is currently the quadruped only (upper body
removed, see UPPER_BODY_MODE in mjcf/build_train_xml.py).
Only the calf meshes collide, so Go2's non-foot "illegal contact" termination is replaced
by `base_too_low` (base collapsing onto its feet) alongside the usual tilt check.
"""

import math
from pathlib import Path

from centaur_rl.assets.robots import get_centaur_robot_cfg
from centaur_rl.assets.robots.centaur.centaur_constants import FOOT_GEOM_NAMES, LEG_NAMES
from centaur_rl.tasks.walking import mdp as centaur_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg

from centaur_rl.tasks.walking.velocity_env_cfg import make_velocity_env_cfg

FOOT_NAMES = ("FR", "FL", "RR", "RL")
SITE_NAMES = FOOT_NAMES
GEOM_NAMES = FOOT_GEOM_NAMES  # tip spheres act as feet
GAIT_JSON = str(Path(__file__).resolve().parents[2] / "mdp" / "gait.json")

# Gait from mdp/gait.json, one stance column per LEG_NAMES entry, turned into trot-style
# clocks: every leg runs the same cycle (swing, then stance for GAIT_STANCE_FRACTION of
# it), and each leg starts it at its own point of the cycle (FR at 0, then RL, FL, RR as
# in the JSON). The JSON fixes the shape of the cycle; its length is the commanded cycle
# time (the JSON's own 1.66 s is the longest), so all its timings scale with it.
GAIT_DURATIONS, GAIT_STANCE_TABLE = centaur_mdp.load_gait_schedule(GAIT_JSON, LEG_NAMES)
(
  GAIT_PERIOD,
  GAIT_STANCE_FRACTION,
  GAIT_LEG_START_TIMES,
  GAIT_AIR_TIMES,
  GAIT_OFFSETS,  # per LEG_NAMES entry, feet_gait convention
) = centaur_mdp.gait_leg_clocks(GAIT_DURATIONS, GAIT_STANCE_TABLE)
# Cycle fraction at which each leg first lifts off when starting from rest.
GAIT_LIFT_PHASES = [t / GAIT_PERIOD for t in GAIT_LEG_START_TIMES]

# Maximum walking speed. A walk vaults over stiff stance legs like an inverted pendulum,
# which only stays on the ground while the centripetal demand v^2 / L stays below g; in
# practice animals leave the walk at a Froude number v^2 / (g L) of about 0.5 (the
# walk-trot transition, Alexander 1989), so v_max = sqrt(0.5 g L).
LEG_LENGTH = 0.58  # m, hip to foot in the default stance
WALK_MAX_FROUDE = 0.5
MAX_WALK_SPEED = math.sqrt(WALK_MAX_FROUDE * 9.81 * LEG_LENGTH)  # ~1.69 m/s
MAX_LIN_VEL_X = min(2.0, MAX_WALK_SPEED)

# Cycle time vs speed. A planted foot sweeps v * stance_fraction * T under its hip, which
# must stay within STANCE_REACH (0.45 m, about +-21 deg of leg swing, without
# over-extending the 0.58 m leg) and should be at least STANCE_REACH_MIN (shorter
# strides are shuffling). So T is in [MIN / (d v), REACH / (d v)]: faster means shorter
# cycles. The shortest cycle is the one that reaches MAX_LIN_VEL_X with a full stride,
# the longest is gait.json's own.
STANCE_REACH = 0.45
STANCE_REACH_MIN = 0.15
CYCLE_TIME_RANGE = (STANCE_REACH / (GAIT_STANCE_FRACTION * MAX_LIN_VEL_X), GAIT_PERIOD)

# Forward speed, no lateral command. The yaw command is not sampled on its own (range 0):
# the walk command sets it to lin_x * curvature, i.e. the robot walks a circle of radius
# 1 / curvature and never turns on the spot (see TURN_* below).
COMMAND_RANGES = {
  "lin_vel_x": (0.0, MAX_LIN_VEL_X),
  "lin_vel_y": (0.0, 0.0),
  "ang_vel_z": (0.0, 0.0),
}

# Turning on circular arcs: yaw rate = lin_x / R. The sharpest turn is R = TURN_RADIUS_MIN,
# further limited at speed by the centripetal acceleration v^2 / R <= TURN_MAX_LAT_ACCEL
# (0.2 g, a lean of ~11 deg, which a walk can hold over its 2-3 foot supports). So up to
# 1.40 m/s the 1 m circle is allowed (yaw up to 1.4 rad/s); at 1.69 m/s R >= 1.45 m.
# Curvature is sampled uniformly in [-1/R_min(v), 1/R_min(v)]; TURN_STRAIGHT_FRAC of the
# walking commands are kept straight. On a turn the outer feet (HALF_TRACK off the
# midline) sweep faster, which the cycle-time range accounts for.
TURN_RADIUS_MIN = 1.0  # m
TURN_MAX_LAT_ACCEL = 0.2 * 9.81  # m/s^2
TURN_STRAIGHT_FRAC = 0.25
HALF_TRACK = 0.23  # m, foot offset from the body midline
TURN_PARAMS = {
  "max_curvature": 1.0 / TURN_RADIUS_MIN,
  "max_lateral_accel": TURN_MAX_LAT_ACCEL,
  "straight_frac": TURN_STRAIGHT_FRAC,
  "half_track": HALF_TRACK,
}

TRACK_LIN_STD = min(0.5, 0.5 * MAX_LIN_VEL_X)
# Yaw rate commands reach ~1.4 rad/s; 0.5 keeps a useful gradient on large yaw errors
# (0.25 gave ~0 reward beyond 0.5 rad/s) while still penalising drift off the arc.
TRACK_ANG_STD = 0.5

# Nominal base-to-foot-site height is 0.60 m (straight legs); below this the robot has collapsed.
MIN_BASE_HEIGHT = 0.30

# Play-only: constant forward speed command (m/s), never standing.
PLAY_LIN_VEL_X = 0.5


def centaur_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Centaur rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_centaur_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = SITE_NAMES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # Knee std loosened (Go2: 0.15) so a slight knee flex is allowed when standing.
  cfg.rewards["pose"].params["std_standing"] = {
    r".*_1": 0.05,
    r".*_2": 0.1,
    r".*_3": 0.3,
  }
  # Knee stds are wider than Go2's: every swing needs a large knee flexion away from the
  # default. Hip ab/ad is wide as well: the gait has two-foot supports (same-side and
  # diagonal pairs) held far longer than the ~0.24 s falling time constant, so the CoM
  # has to sit over the support line. The feet are 0.23 m off the midline, which takes
  # up to ~0.4 rad of ab/ad (leaning over, or stepping toward the midline).
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_1": 0.4,
    r".*_2": 0.35,
    r".*_3": 0.8,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_1": 0.4,
    r".*_2": 0.5,
    r".*_3": 1.0,
  }

  # Walk command: forward speed along a circular arc (yaw = lin_x * curvature) plus a gait
  # cycle time (4th command entry, so the policy observes it), drawn from the range
  # feasible for the speed and turn. The command term also runs the gait clock,
  # restarted from rest on every standing -> walking switch.
  cfg.commands["twist"] = centaur_mdp.WalkVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.05,
    heading_command=False,
    debug_vis=True,
    ranges=centaur_mdp.WalkVelocityCommandCfg.Ranges(
      **COMMAND_RANGES, cycle_time=CYCLE_TIME_RANGE
    ),
    stance_fraction=GAIT_STANCE_FRACTION,
    stance_reach=(STANCE_REACH_MIN, STANCE_REACH),
    **TURN_PARAMS,
  )
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, **COMMAND_RANGES, "lin_vel_x": (0.0, 0.5 * MAX_LIN_VEL_X)},
    {"step": 5000 * 24, **COMMAND_RANGES},
  ]

  # Walk rewarded like the trot (feet_gait: fraction of feet matching their clock), with
  # the clock from the command and each leg held on the ground until its first lift-off
  # when starting from rest (FR's long first stride, then RL, FL, RR, as gait.json P0).
  gait_clock_params = {
    "offset": GAIT_OFFSETS,
    "lift_phase": GAIT_LIFT_PHASES,
    "threshold": GAIT_STANCE_FRACTION,
    "foot_geoms": GEOM_NAMES,
    "command_name": "twist",
    "sensor_name": "feet_ground_contact",
  }
  # Gait accuracy over robustness. Half of the gait reward is only paid while all four
  # feet match: a best-aligned trot matches 75% of the walk's foot states (so with the
  # per-foot fraction alone a walk earns just 0.25 more than a trot), but all four feet
  # only 45% of the time. Per step a perfect walk earns 1.5 vs ~0.9 for a trot.
  cfg.rewards["foot_gait"].func = centaur_mdp.feet_gait_clock
  cfg.rewards["foot_gait"].weight = 1.5
  cfg.rewards["foot_gait"].params = {**gait_clock_params, "exact_weight": 0.5}
  # Slight penalty for a foot on the ground while its clock says swing.
  cfg.rewards["foot_contact_in_swing"] = RewardTermCfg(
    func=centaur_mdp.feet_contact_in_swing, weight=-0.5, params=dict(gait_clock_params)
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["phase"] = ObservationTermCfg(
      func=centaur_mdp.walk_phase, params={"command_name": "twist"}
    )
  cfg.rewards["track_linear_velocity"].params["std"] = TRACK_LIN_STD
  cfg.rewards["track_angular_velocity"].params["std"] = TRACK_ANG_STD

  # Pushes scaled to the gait speed (no-op once the top speed is >= 0.5 m/s): kicks well
  # above the walking speed can only be survived by putting a foot down off-schedule,
  # which teaches the policy to hedge instead of following it.
  push_scale = min(1.0, 2.0 * MAX_LIN_VEL_X)
  push_range = cfg.events["push_robot"].params["velocity_range"]
  for axis, (lo, hi) in push_range.items():
    push_range[axis] = (push_scale * lo, push_scale * hi)

  # Roll and pitch penalised equally (tilt and tilt rate). Roll is measured from the
  # equilibrium lean of the current turn, theta = atan(v^2 / (r g)) with the measured
  # forward speed v and the commanded radius r = 1 / curvature (upright when straight).
  # The steady lean has zero roll rate, so the rate term needs no target.
  cfg.rewards["body_orientation_l2"].func = centaur_mdp.body_orientation_roll_pitch
  cfg.rewards["body_orientation_l2"].params = {
    "roll_weight": 1.0,
    "pitch_weight": 1.0,
    "command_name": "twist",
    "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
  }
  cfg.rewards["body_ang_vel"].func = centaur_mdp.body_angular_velocity_roll_pitch
  cfg.rewards["body_ang_vel"].params = {
    "roll_weight": 1.0,
    "pitch_weight": 1.0,
    "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
  }
  cfg.rewards["foot_clearance"].weight = -0.3
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = SITE_NAMES
  cfg.rewards["foot_clearance"].params["target_height"] = 0.13  # m swing clearance
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = SITE_NAMES

  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=centaur_mdp.base_too_low,
    params={
      "minimum_height": MIN_BASE_HEIGHT,
      "asset_cfg": SceneEntityCfg("robot", site_names=SITE_NAMES),
    },
  )

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def centaur_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Centaur flat terrain velocity configuration."""
  cfg = centaur_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    # Play-only: constant forward speed. Training is unaffected.
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, centaur_mdp.WalkVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (PLAY_LIN_VEL_X, PLAY_LIN_VEL_X)
    twist_cmd.max_curvature = 0.0
    twist_cmd.rel_standing_envs = 0.0

  return cfg


##
# SGS (Success-Guided Sampling, arXiv:2610.12465) variants.
##

# Paper's locomotion hyper-parameters (Sec. 3.2 / App. B.2).
SGS_PARAMS = {
  "target": 0.66,
  "kappa": 5.0,
  "temperature": 2.0,
  "eps": 1e-8,
  "history": 100,
  # Success test for one episode (velocity-tracking analogue of "command-success").
  # Thresholds scale with the gait's speed envelope, otherwise standing still would
  # count as tracking a slow gait's commands.
  "max_err_xy": min(0.3, 0.4 * MAX_LIN_VEL_X),  # m/s, episode-mean |v_cmd_xy - v_xy|
  "max_err_yaw": 0.2,  # rad/s, episode-mean |w_z_cmd - w_z| (yaw = lin_x / R)
  "min_episode_frac": 0.5,  # ignore time-outs of truncated (randomized-start) episodes
  "standing_frac": 0.05,
  "seed": 0,
}
# Each SGS command config is (lin_x, 0, lin_x * curvature, cycle time). The turn
# sharpness (curvature = 1 / radius, TURN_PARAMS) is a per-config parameter drawn from
# the range feasible for its speed; the cycle time is drawn per config from the range
# feasible for its speed and turn (faster -> shorter).
SGS_COMMAND_RANGES = {**COMMAND_RANGES, "cycle_time": CYCLE_TIME_RANGE}


def apply_sgs(
  cfg: ManagerBasedRlEnvCfg, num_spawns: int, num_commands: int
) -> ManagerBasedRlEnvCfg:
  """Replace the terrain/command curricula with SGS over (tile, spawn, command) configs."""
  hold = cfg.episode_length_s + 1.0  # command fixed for the whole episode
  cfg.commands["twist"] = centaur_mdp.SgsVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(hold, hold),
    rel_standing_envs=0.0,
    heading_command=False,
    debug_vis=True,
    ranges=centaur_mdp.SgsVelocityCommandCfg.Ranges(**SGS_COMMAND_RANGES),
    stance_fraction=GAIT_STANCE_FRACTION,
    stance_reach=(STANCE_REACH_MIN, STANCE_REACH),
    **TURN_PARAMS,
  )
  cfg.events["reset_base"] = EventTermCfg(func=centaur_mdp.reset_root_state_sgs, mode="reset")
  cfg.curriculum = {
    "sgs": CurriculumTermCfg(
      func=centaur_mdp.SuccessGuidedSampling,
      params={
        **SGS_PARAMS,
        "command_name": "twist",
        "num_spawns": num_spawns,
        "num_commands": num_commands,
        "command_ranges": SGS_COMMAND_RANGES,
        "stance_fraction": GAIT_STANCE_FRACTION,
        "stance_reach": (STANCE_REACH_MIN, STANCE_REACH),
        **TURN_PARAMS,
      },
    )
  }
  return cfg


# Config-set size N. The paper uses N=104k with 4K-1M parallel envs. At ~1K envs and
# 20 s episodes a full run is only ~250k episodes, so 104k configs would each be seen
# ~2x and never get usable success estimates. Scale N up with num_envs (paper: 104k).
def centaur_rough_sgs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # 10 x 20 tiles x 4 spawns x 32 commands = 25,600 configs (32 commands to cover
  # speed x turn sharpness x cycle time).
  cfg = centaur_rough_env_cfg(play=play)
  return cfg if play else apply_sgs(cfg, num_spawns=4, num_commands=32)

