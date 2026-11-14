"""Centaur pace environment configurations.

Same setup as the walking task (centaur_rl/tasks/walking); only the gait differs: a pace
(the two right legs together, then the two left legs) with a flight phase in between.

Same paradigm as the Unitree Go2 velocity task (unitree_rl_mjlab): 12 leg joints are the
action/observation space. The training MJCF is currently the quadruped only (upper body
removed, see UPPER_BODY_MODE in mjcf/build_train_xml.py).
Only the calf meshes collide, so Go2's non-foot "illegal contact" termination is replaced
by `base_too_low` (base collapsing onto its feet) alongside the usual tilt check.
"""

import math

from centaur_rl.assets.robots import get_centaur_robot_cfg
from centaur_rl.assets.robots.centaur.centaur_constants import FOOT_GEOM_NAMES, LEG_NAMES
from centaur_rl.tasks.pace import mdp as centaur_mdp
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

from centaur_rl.tasks.pace.velocity_env_cfg import make_velocity_env_cfg

FOOT_NAMES = ("FR", "FL", "RR", "RL")
SITE_NAMES = FOOT_NAMES
GEOM_NAMES = FOOT_GEOM_NAMES  # tip spheres act as feet
assert LEG_NAMES == FOOT_NAMES  # the per-leg gait lists below are in this order

# Pace with flight phases, on the same clock as the walking task. Cycle time is fixed to
# the trot's (velocity task). One cycle, starting from rest at cycle 0:
#   right pair (FR + RR) in the air, left pair (FL + RL) on the ground
#   flight: all four feet in the air
#   left pair (FL + RL) in the air, right pair (FR + RR) on the ground
#   flight: all four feet in the air
# A leg is in stance while (cycles + offset) % 1 < 0.5 - flight fraction(speed). One
# flight phase lasts FLIGHT_FRACTION[0] of the cycle at FLIGHT_SPEED_RANGE[0] m/s
# commanded speed, growing linearly to FLIGHT_FRACTION[1] at FLIGHT_SPEED_RANGE[1] (the
# same mapping as the velocity task's flying trot):
#   0.5 m/s:  0.05 of the cycle = 0.030 s flight, foot air time 0.330 s
#   1.0 m/s:  0.08 of the cycle = 0.048 s flight, foot air time 0.348 s
#   1.69 m/s: 0.12 of the cycle = 0.073 s flight, foot air time 0.373 s
GAIT_PERIOD = 0.6  # s, the velocity task's trot period
GAIT_STANCE_FRACTION = 0.5  # before the flight fraction is taken off
GAIT_OFFSETS = [0.5, 0.0, 0.5, 0.0]  # FR, FL, RR, RL: left pair half a cycle off the right
GAIT_LIFT_PHASES = [0.0, 0.0, 0.0, 0.0]  # no startup hold: the right pair lifts at cycle 0
FLIGHT_FRACTION = (0.02, 0.14)
FLIGHT_SPEED_RANGE = (0.0, 2.0)  # m/s

# Speed envelope kept from the walking task, whose reasoning was: a walk vaults over
# stiff stance legs like an inverted pendulum,
# which only stays on the ground while the centripetal demand v^2 / L stays below g; in
# practice animals leave the walk at a Froude number v^2 / (g L) of about 0.5 (the
# walk-trot transition, Alexander 1989), so v_max = sqrt(0.5 g L).
LEG_LENGTH = 0.58  # m, hip to foot in the default stance
WALK_MAX_FROUDE = 0.5
MAX_WALK_SPEED = math.sqrt(WALK_MAX_FROUDE * 9.81 * LEG_LENGTH)  # ~1.69 m/s
MAX_LIN_VEL_X = min(2.0, MAX_WALK_SPEED)

# Fixed cycle time (the command still carries it as its 4th entry, as in the walking
# task). At MAX_LIN_VEL_X the stance foot sweeps v * stance fraction * T = 0.38 m under
# its hip, within STANCE_REACH.
STANCE_REACH = 0.45
STANCE_REACH_MIN = 0.15
CYCLE_TIME_RANGE = (GAIT_PERIOD, GAIT_PERIOD)

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
PLAY_LIN_VEL_X = 0.3


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

  # Pace rewarded on the command's gait clock (feet_gait_clock: fraction of feet matching
  # their schedule), with the speed-dependent flight phases.
  gait_clock_params = {
    "offset": GAIT_OFFSETS,
    "lift_phase": GAIT_LIFT_PHASES,
    "threshold": GAIT_STANCE_FRACTION,
    "flight_fraction": FLIGHT_FRACTION,
    "flight_speed_range": FLIGHT_SPEED_RANGE,
    "foot_geoms": GEOM_NAMES,
    "command_name": "twist",
    "sensor_name": "feet_ground_contact",
  }
  # Half of the gait reward is only paid while all four feet match: a trot matches half
  # of the pace's foot states at any time, but never all four.
  cfg.rewards["foot_gait"].func = centaur_mdp.feet_gait_clock
  cfg.rewards["foot_gait"].weight = 1.5
  cfg.rewards["foot_gait"].params = {**gait_clock_params, "exact_weight": 0.5}
  # Slight penalty for a foot on the ground while its clock says swing.
  cfg.rewards["foot_contact_in_swing"] = RewardTermCfg(
    func=centaur_mdp.feet_contact_in_swing, weight=-0.5, params=dict(gait_clock_params)
  )
  # Slight reward for the suspension: paid while the clock is in a flight phase and all
  # four feet are off the ground. Averages 2 * flight fraction per step when achieved
  # (0.16 at 1 m/s, 0.24 at top speed), i.e. at most +0.05 / +0.07 per step at this weight.
  cfg.rewards["foot_suspension"] = RewardTermCfg(
    func=centaur_mdp.feet_suspension, weight=0.3, params=dict(gait_clock_params)
  )
  # The flight phases need the base to bounce, which the default z-velocity weight of 2
  # would cost a large part of the tracking reward.
  cfg.rewards["track_linear_velocity"].params["z_weight"] = 0.5
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
  # Success test (see mdp/sgs.py): the mean velocity while on the config's tile, after
  # the first second, is within max(0.15, 0.25 v) m/s of the commanded speed v and
  # 0.2 rad/s of the commanded yaw rate (v / R), and the robot did not fall on the tile.
  # The speed tolerance is relative so that standing still does not count as tracking a
  # slow pace (0.15 m/s at the slowest command, 0.42 m/s at top speed).
  "max_err_xy": 0.15,
  "max_err_xy_rel": 0.25,
  "max_err_yaw": 0.2,
  "min_episode_frac": 0.5,  # ignore time-outs of truncated (randomized-start) episodes
  # Share of episodes that are standing episodes outside the config set (as the base
  # task's rel_standing_envs), so standing is practised whatever SGS samples.
  "standing_prob": 0.05,
}
SGS_SETTLE_TIME = 1.0  # s after reset left out of the success test (starting from rest)
# Slowest SGS pace command. Below it a command is hard to tell from standing, which is
# covered by standing_prob instead.
SGS_MIN_LIN_VEL_X = 0.3


def sgs_axes(speed_bins: int, turn_bins: int) -> dict:
  """SGS command axes, (low, high, bins) each.

  lin_vel_x: forward speed. turn: turn sharpness in [-1, 1], curvature = turn * k_max(v)
  with k_max = min(1 / TURN_RADIUS_MIN, TURN_MAX_LAT_ACCEL / v^2), yaw rate = v *
  curvature: the pace only turns along arcs, never on the spot. turn_bins is odd and the
  middle bin is exactly straight.
  """
  return {
    "lin_vel_x": (SGS_MIN_LIN_VEL_X, MAX_LIN_VEL_X, speed_bins),
    "turn": (-1.0, 1.0, turn_bins),
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
    ranges=centaur_mdp.SgsVelocityCommandCfg.Ranges(
      **COMMAND_RANGES, cycle_time=CYCLE_TIME_RANGE
    ),
    stance_fraction=GAIT_STANCE_FRACTION,
    stance_reach=(STANCE_REACH_MIN, STANCE_REACH),
    settle_time=SGS_SETTLE_TIME,
    **TURN_PARAMS,
  )
  cfg.events["reset_base"] = EventTermCfg(func=centaur_mdp.reset_root_state_sgs, mode="reset")
  cfg.curriculum = {
    "sgs": CurriculumTermCfg(
      func=centaur_mdp.SuccessGuidedSampling,
      params={
        **SGS_PARAMS,
        "command_name": "twist",
        "axes": axes,
        "snap_center": ("turn",),
      },
    )
  }
  return cfg


# Config-set size N. The paper uses N=104k with 4K-1M parallel envs. A run at ~1K envs is
# about 250k episodes (10k iterations x 24 steps x 0.02 s x 1024 envs / 20 s), and a
# config needs tens of outcomes before its success rate means anything (H = 100), so N
# is kept to a few thousand: ~40 episodes per config on average, more on the frontier.
# Raise the bin counts along with num_envs.
def centaur_rough_sgs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # 10 levels x 20 terrain columns x (6 speeds x 5 turns) = 6,000 configs.
  cfg = centaur_rough_env_cfg(play=play)
  return cfg if play else apply_sgs(cfg, sgs_axes(speed_bins=6, turn_bins=5))

