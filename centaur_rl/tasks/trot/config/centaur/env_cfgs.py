"""Centaur velocity environment configurations.

Same paradigm as the Unitree Go2 velocity task (unitree_rl_mjlab): 12 leg joints are the
action/observation space. The training MJCF is currently the quadruped only (upper body
removed, see UPPER_BODY_MODE in mjcf/build_train_xml.py).
Only the calf meshes collide, so Go2's non-foot "illegal contact" termination is replaced
by `base_too_low` (base collapsing onto its feet) alongside the usual tilt check.
"""

from centaur_rl.assets.robots import get_centaur_robot_cfg
from centaur_rl.assets.robots.centaur.centaur_constants import FOOT_GEOM_NAMES
from centaur_rl.tasks.trot import mdp as centaur_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from centaur_rl.tasks.trot.velocity_env_cfg import make_velocity_env_cfg

FOOT_NAMES = ("FR", "FL", "RR", "RL")
SITE_NAMES = FOOT_NAMES
GEOM_NAMES = FOOT_GEOM_NAMES  # tip spheres act as feet

# Flying trot. Each diagonal pair's stance is followed by a flight phase with all four
# feet in the air (there is no double-support overlap). One flight phase lasts
# FLIGHT_FRACTION[0] of the cycle at FLIGHT_SPEED_RANGE[0] m/s commanded translational
# speed, growing linearly to FLIGHT_FRACTION[1] at FLIGHT_SPEED_RANGE[1]:
#   1 m/s: 0.08 of the cycle = 0.048 s flight, foot air time 0.348 s, stance 0.252 s
#   2 m/s: 0.14 of the cycle = 0.084 s flight, foot air time 0.384 s, stance 0.216 s
# (0.084 s of flight is a 0.41 m/s take-off and ~1 cm of apex height; the stance foot
# sweeps 0.43 m under the hip at 2 m/s, within the 0.58 m leg's reach).
GAIT_PERIOD = 0.6  # s
FLIGHT_FRACTION = (0.02, 0.14)
FLIGHT_SPEED_RANGE = (0.0, 2.0)  # m/s

# Forward speed is always commanded above 1 m/s: no standing, no backward.
MIN_LIN_VEL_X = 1.0
MAX_LIN_VEL_X = 2.0
COMMAND_RANGES = {
  "lin_vel_x": (MIN_LIN_VEL_X, MAX_LIN_VEL_X),
  "lin_vel_y": (-0.5, 0.5),
  "ang_vel_z": (-1.0, 1.0),
}

# Nominal base-to-foot-site height is 0.60 m (straight legs); below this the robot has collapsed.
MIN_BASE_HEIGHT = 0.30


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
  # Knee stds are wider than Go2's: with a straight-leg default, every swing (0.15 m
  # clearance needs ~1 rad of knee flexion) is a large one-sided deviation from default.
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_1": 0.15,
    r".*_2": 0.35,
    r".*_3": 0.8,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_1": 0.15,
    r".*_2": 0.5,
    r".*_3": 1.0,
  }

  # Commands: lin_x always in [MIN_LIN_VEL_X, MAX_LIN_VEL_X], never standing.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.ranges.lin_vel_x = COMMAND_RANGES["lin_vel_x"]
  twist_cmd.ranges.lin_vel_y = COMMAND_RANGES["lin_vel_y"]
  twist_cmd.ranges.ang_vel_z = COMMAND_RANGES["ang_vel_z"]
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, **COMMAND_RANGES, "lin_vel_x": (MIN_LIN_VEL_X, 1.5), "lin_vel_y": (-0.3, 0.3)},
    {"step": 5000 * 24, **COMMAND_RANGES},
  ]

  # Flying trot: FR+RL in phase, FL+RR half a period later (order matches FOOT_NAMES).
  # Stance fraction = 0.5 - flight fraction(speed), so between the two pairs' stances
  # every foot is scheduled in the air.
  gait_params = cfg.rewards["foot_gait"].params
  gait_params["period"] = GAIT_PERIOD
  gait_params["offset"] = [0.0, 0.5, 0.5, 0.0]
  gait_params["threshold"] = 0.5
  gait_params["flight_fraction"] = FLIGHT_FRACTION
  gait_params["flight_speed_range"] = FLIGHT_SPEED_RANGE
  # Slight reward for the suspension: paid while the schedule is in a flight phase and all
  # four feet are off the ground. Averages 2 * flight fraction per step when achieved
  # (0.16 at 1 m/s, 0.28 at 2 m/s), i.e. at most +0.05 / +0.08 per step at this weight.
  cfg.rewards["foot_suspension"] = RewardTermCfg(
    func=centaur_mdp.feet_suspension, weight=0.3, params=dict(gait_params)
  )
  cfg.observations["actor"].terms["phase"].params["period"] = GAIT_PERIOD
  # The flight phases need the base to bounce (up to ~0.4 m/s vertically at 2 m/s), which
  # the default z-velocity weight of 2 would cost about half of the tracking reward.
  cfg.rewards["track_linear_velocity"].params["z_weight"] = 0.5
  # Commands are always >= 1 m/s: use the running pose tolerance for all of them.
  cfg.rewards["pose"].params["running_threshold"] = MIN_LIN_VEL_X
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = SITE_NAMES
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
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = COMMAND_RANGES["lin_vel_x"]
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

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
  # Success test (see mdp/sgs.py): the mean velocity while on the config's tile, after
  # the first second, is within max(0.3, 0.2 |cmd_xy|) m/s (0.3 up to 1.5 m/s, 0.4 at
  # 2 m/s) and 0.3 rad/s of the command, and the robot did not fall on the tile.
  "max_err_xy": 0.3,
  "max_err_xy_rel": 0.2,
  "max_err_yaw": 0.3,
  "min_episode_frac": 0.5,  # ignore time-outs of truncated (randomized-start) episodes
  "standing_prob": 0.0,  # lin_x is always >= MIN_LIN_VEL_X
}
SGS_SETTLE_TIME = 1.0  # s after reset left out of the success test (0 -> 1-2 m/s)


def sgs_axes(lin_x_bins: int, lin_y_bins: int, yaw_bins: int) -> dict:
  """SGS command axes: the commanded velocity, (low, high, bins) per component."""
  return {
    "lin_vel_x": (*COMMAND_RANGES["lin_vel_x"], lin_x_bins),
    "lin_vel_y": (*COMMAND_RANGES["lin_vel_y"], lin_y_bins),
    "ang_vel_z": (*COMMAND_RANGES["ang_vel_z"], yaw_bins),
  }


def apply_sgs(cfg: ManagerBasedRlEnvCfg, axes: dict) -> ManagerBasedRlEnvCfg:
  """Replace the terrain/command curricula with SGS over (tile, command cell) configs."""
  hold = cfg.episode_length_s + 1.0  # command fixed for the whole episode
  cfg.commands["twist"] = centaur_mdp.SgsVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(hold, hold),
    rel_standing_envs=0.0,
    heading_command=False,
    debug_vis=True,
    ranges=centaur_mdp.SgsVelocityCommandCfg.Ranges(**COMMAND_RANGES),
    settle_time=SGS_SETTLE_TIME,
  )
  cfg.events["reset_base"] = EventTermCfg(func=centaur_mdp.reset_root_state_sgs, mode="reset")
  cfg.curriculum = {
    "sgs": CurriculumTermCfg(
      func=centaur_mdp.SuccessGuidedSampling,
      params={**SGS_PARAMS, "command_name": "twist", "axes": axes},
    )
  }
  return cfg


# Config-set size N. The paper uses N=104k with 4K-1M parallel envs. A run at ~1K envs is
# about 250k episodes (10k iterations x 24 steps x 0.02 s x 1024 envs / 20 s), and a
# config needs tens of outcomes before its success rate means anything (H = 100), so N
# is kept to a few thousand: ~35 episodes per config on average, more on the frontier.
# Raise the bin counts along with num_envs.
def centaur_rough_sgs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # 10 levels x 20 terrain columns x (4 lin_x x 3 lin_y x 3 yaw) = 7,200 configs.
  cfg = centaur_rough_env_cfg(play=play)
  return cfg if play else apply_sgs(cfg, sgs_axes(lin_x_bins=4, lin_y_bins=3, yaw_bins=3))

