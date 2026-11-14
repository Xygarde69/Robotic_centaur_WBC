"""Centaur velocity environment configurations.

Same paradigm as the Unitree Go2 velocity task (unitree_rl_mjlab): 12 leg joints are the
action/observation space. The training MJCF is currently the quadruped only (upper body
removed, see UPPER_BODY_MODE in mjcf/build_train_xml.py).
Only the calf meshes collide, so Go2's non-foot "illegal contact" termination is replaced
by `base_too_low` (base collapsing onto its feet) alongside the usual tilt check.
"""

from centaur_rl.assets.robots import get_centaur_robot_cfg
from centaur_rl.assets.robots.centaur.centaur_constants import FOOT_GEOM_NAMES
from centaur_rl.tasks.velocity import mdp as centaur_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from centaur_rl.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

FOOT_NAMES = ("FR", "FL", "RR", "RL")
SITE_NAMES = FOOT_NAMES
GEOM_NAMES = FOOT_GEOM_NAMES  # tip spheres act as feet

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

  # Trot: FR+RL in phase, FL+RR half a period later (order matches FOOT_NAMES).
  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
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
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
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
  # Success test for one episode (velocity-tracking analogue of "command-success").
  "max_err_xy": 0.3,  # m/s, episode-mean |v_cmd_xy - v_xy|
  "max_err_yaw": 0.3,  # rad/s, episode-mean |w_cmd_z - w_z|
  "min_episode_frac": 0.5,  # ignore time-outs of truncated (randomized-start) episodes
  "standing_frac": 0.05,
  "seed": 0,
}
SGS_COMMAND_RANGES = {
  "lin_vel_x": (-1.0, 2.0),
  "lin_vel_y": (-1.0, 1.0),
  "ang_vel_z": (-1.0, 1.0),
}


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
      },
    )
  }
  return cfg


# Config-set size N. The paper uses N=104k with 4K-1M parallel envs. At ~1K envs and
# 20 s episodes a full run is only ~250k episodes, so 104k configs would each be seen
# ~2x and never get usable success estimates. Scale N up with num_envs (paper: 104k).
def centaur_rough_sgs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # 10 x 20 tiles x 4 spawns x 16 commands = 12,800 configs.
  cfg = centaur_rough_env_cfg(play=play)
  return cfg if play else apply_sgs(cfg, num_spawns=4, num_commands=16)


def centaur_flat_sgs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # No tiles: 4 spawns x 1024 commands = 4,096 configs.
  cfg = centaur_flat_env_cfg(play=play)
  return cfg if play else apply_sgs(cfg, num_spawns=4, num_commands=1024)
