"""Env config for one (gait, phase) stage: straight line, flat ground, speed SGS.

make_env_cfg(spec, phase) is the only factory; every stage task is registered from it
(config/centaur/__init__.py). What differs between stages:

  gait   clock, speed range, stride frequency, spawn, terminations, rewards (GaitSpec +
         rewards/<gait>.py)
  phase  ladder   SpeedSGS "ladder" (V_current climbs), command fixed per episode, straight
         refine   SpeedSGS "refine" (whole speed range x turn radius 2 km -> R_min),
                  non-holonomic arcs ang_z = lin_x_cmd / r, speed resampled mid-episode
         handoff  as refine but straight, and a large share of resets start from banks

What NEVER differs (each gait starts from the previous gait's weights): observation
layout, action space, action scale, PD gains, torque limit. The control rate is per gait
(GaitSpec.decimation): 50 Hz for walk/trot/canter, 100 Hz for the gallop, whose 60-100 ms
stances are only 3-5 steps at 50 Hz. The gallop forks from the 50 Hz canter: the obs
layout is identical, but the policy now acts every 10 ms (critic warmup + std floor
absorb the change).
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from centaur_rl.assets.robots import get_centaur_robot_cfg
from centaur_rl.assets.robots.centaur.centaur_constants import FOOT_GEOM_NAMES, LEG_NAMES
from centaur_rl.tasks.Multi_Policy import mdp
from centaur_rl.tasks.Multi_Policy.gaits import NUM_GAITS, SIM_DT, GaitSpec
from centaur_rl.tasks.Multi_Policy.pipeline import LADDER_ADVANCE_THRESHOLD, LADDER_MIN_OUTCOMES
from centaur_rl.tasks.Multi_Policy.rewards import build_rewards
from centaur_rl.tasks.Multi_Policy.rewards.common import COMMAND, CONTACT_SENSOR, FOOT_SITES, RAMPED_PENALTIES

PHASES = ("ladder", "refine", "handoff")

##
# Fixed for every gait (changing any of these invalidates every checkpoint).
##

# Sim-only robot (never goes to hardware): torque limit raised so it is not what caps
# the gallop. 120 Nm was the motor's real peak. Check Metrics/peak_torque per gait.
EFFORT_LIMIT = 400.0  # Nm
# Joint-position action scale, sized for the gallop's joint excursions (the walking task
# used 0.25). Fixed from the walk on because it sets what the network's outputs mean.
ACTION_SCALE = 0.5
EPISODE_LENGTH_S = 20.0


def robot_cfg():
  cfg = get_centaur_robot_cfg()
  art = cfg.articulation
  assert art is not None
  actuators = tuple(replace(a, effort_limit=EFFORT_LIMIT) for a in art.actuators)
  cfg.articulation = replace(art, actuators=actuators)
  return cfg


def _observations() -> dict[str, ObservationGroupCfg]:
  cmd = {"command_name": COMMAND}
  actor = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "gait_command": ObservationTermCfg(func=mdp.gait_command, params=cmd),
    "gait_clock": ObservationTermCfg(func=mdp.gait_clock, params=cmd),
    "gait_onehot": ObservationTermCfg(func=mdp.gait_onehot, params=cmd),
    "gait_duty": ObservationTermCfg(func=mdp.gait_duty, params=cmd),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  feet_sites = SceneEntityCfg("robot", site_names=FOOT_SITES)
  sensor = {"sensor_name": CONTACT_SENSOR}
  critic = {
    **{k: replace(v, noise=None) for k, v in actor.items()},
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "gait_stance": ObservationTermCfg(func=mdp.gait_stance, params=cmd),
    "foot_height": ObservationTermCfg(func=mdp.foot_height, params={"asset_cfg": feet_sites}),
    "foot_air_time": ObservationTermCfg(func=mdp.foot_air_time, params=sensor),
    "foot_contact": ObservationTermCfg(func=mdp.foot_contact, params=sensor),
    "foot_contact_forces": ObservationTermCfg(func=mdp.foot_contact_forces, params=sensor),
  }
  return {
    "actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=True),
    "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
  }


def _command(spec: GaitSpec, phase: str, play: bool) -> mdp.GaitCommandCfg:
  held = phase == "ladder" and not play
  hold = EPISODE_LENGTH_S + 1.0
  return mdp.GaitCommandCfg(
    entity_name="robot",
    command_name=COMMAND,
    resampling_time_range=(hold, hold) if held else spec.refine_resample_time,
    resample_mid_episode=not held,
    debug_vis=play,
    gait_name=spec.name,
    gait_index=spec.index,
    num_gaits=NUM_GAITS,
    speed_range=spec.sgs_speed_range,
    foot_geoms=FOOT_GEOM_NAMES,
    offsets=spec.clock.offsets,
    duty_factor=spec.clock.duty_factor,
    air_time=spec.clock.air_time,
    duty_min=spec.clock.duty_min,
    duty_max=spec.clock.duty_max,
    stance_sweep=spec.stride.stance_sweep,
    f_min=spec.stride.f_min,
    f_max=spec.stride.f_max,
    f_slope=spec.stride.slope,
    f_offset=spec.stride.offset,
    track_footfalls=spec.success_timing_tol is not None,
    contact_sensor=CONTACT_SENSOR,
    random_start_phase=spec.random_start_phase,
    settle_time=spec.settle_time,
    # Turns in the refine stage only (ladder and handoff stay straight).
    turns=phase == "refine",
    turn_radii=spec.turn_radii,
    turn_max_lat_accel=spec.turn_max_lat_accel,
  )


def _events(spec: GaitSpec, phase: str) -> dict[str, EventTermCfg]:
  bank_prob = {"ladder": 0.0, "refine": spec.refine_handoff_prob, "handoff": spec.handoff_stage_prob}[phase]
  return {
    "reset_gait_state": EventTermCfg(
      func=mdp.GaitStateReset,
      mode="reset",
      params={
        "command_name": COMMAND,
        "spawn_from_rest": spec.spawn_from_rest,
        "joint_noise": spec.spawn_joint_noise,
        "bank_paths": (),  # filled per stage by the runner (pipeline)
        "bank_prob": bank_prob,
      },
    ),
    # Mild domain randomisation (sim only, but keeps the gaits from relying on exact
    # friction / mass distribution).
    "foot_friction": EventTermCfg(
      mode="startup", func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
        "operation": "abs", "ranges": (0.6, 1.2), "shared_random": True,
      },
    ),
    "base_com": EventTermCfg(
      mode="startup", func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "add", "ranges": {0: (-0.03, 0.03), 1: (-0.03, 0.03), 2: (-0.03, 0.03)},
      },
    ),
  }


def _terminations(spec: GaitSpec) -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation, params={"limit_angle": math.radians(spec.max_tilt_deg)}
    ),
    "base_too_low": TerminationTermCfg(
      func=mdp.base_too_low,
      params={
        "minimum_height": spec.min_base_height,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
      },
    ),
  }


def _metrics() -> dict[str, MetricsTermCfg]:
  cs = {"command_name": COMMAND, "sensor_name": CONTACT_SENSOR}
  s = {"sensor_name": CONTACT_SENSOR}
  return {
    "gait_foot_match": MetricsTermCfg(func=mdp.gait_foot_match, params=cs),
    "gait_exact_match": MetricsTermCfg(func=mdp.gait_exact_match, params=cs),
    "duty_factor": MetricsTermCfg(func=mdp.duty_factor, params=s),
    "flight_phase": MetricsTermCfg(func=mdp.flight_phase, params=s),
    "footfall_timing_error": MetricsTermCfg(func=mdp.footfall_timing_error, params={"command_name": COMMAND}),
    "forward_speed_error": MetricsTermCfg(func=mdp.forward_speed_error, params={"command_name": COMMAND}),
    "peak_torque": MetricsTermCfg(func=mdp.peak_torque),
    "mechanical_power": MetricsTermCfg(func=mdp.mechanical_power),
    "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
  }


def _sgs(spec: GaitSpec, phase: str) -> CurriculumTermCfg:
  return CurriculumTermCfg(
    func=mdp.SpeedSGS,
    params={
      "command_name": COMMAND,
      "mode": "ladder" if phase == "ladder" else "refine",
      "speed_range": spec.sgs_speed_range,
      "num_bins": spec.speed_bins,
      # Paper's locomotion hyper-parameters.
      "target": 0.66, "kappa": 5.0, "temperature": 2.0, "eps": 1e-8, "history": 100,
      # Success test.
      "err_abs": spec.success_err_abs,
      "err_rel": spec.success_err_rel,
      "err_yaw": spec.success_err_yaw,
      # Footfall timing is logged (Metrics/footfall_timing_error) but NOT part of the
      # success test: in run gaits_v2 the walk refine's error sat at ~0.26 against a 0.05
      # tolerance, so every episode failed, the radius frontier never left the widest
      # radius and the policy never saw a turn. The clock rewards still shape footfalls.
      "timing_tol": None,
      "err_yaw_rel": 0.2,  # turns: |mean yaw-rate error| <= max(err_yaw, 0.2 |w_cmd|)
      "min_counted_time": 2.0,
      # Turn-radius axis (refine only): per-speed radius frontier, widest radius first.
      "turn_radii": spec.turn_radii if phase == "refine" else (),
      "turn_max_lat_accel": spec.turn_max_lat_accel,
      "refine_frontier_prob": 0.5,
      # Ladder: mostly at V_current, the rest replays slower mastered speeds.
      "frontier_prob": 0.8,
      "advance_threshold": LADDER_ADVANCE_THRESHOLD,
      "retreat_threshold": 0.0,
      "min_outcomes": LADDER_MIN_OUTCOMES,
    },
  )


def _curriculum(spec: GaitSpec, phase: str) -> dict[str, CurriculumTermCfg]:
  terms = {"sgs": _sgs(spec, phase)}
  iters = spec.budget.penalty_ramp_iterations
  if iters > 0 and phase == "refine":
    steps_per_iteration = round(24 * 0.02 / spec.control_dt)  # rl/cfg.py NUM_STEPS_PER_ENV rescaled
    terms["penalty_ramp"] = CurriculumTermCfg(
      func=mdp.PenaltyRamp,
      params={"terms": RAMPED_PENALTIES, "start_scale": 0.1, "ramp_steps": iters * steps_per_iteration},
    )
  return terms


def make_env_cfg(spec: GaitSpec, phase: str = "ladder", play: bool = False) -> ManagerBasedRlEnvCfg:
  assert phase in PHASES, phase
  feet_contact = ContactSensorCfg(
    name=CONTACT_SENSOR,
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    # Every physics substep of the last policy step: the gallop's impulse / peak-force
    # terms integrate it (stance is only a few policy steps long).
    history_length=spec.decimation,
  )
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg()},
      sensors=(feet_contact,),
      num_envs=1,
      extent=2.0,
    ),
    observations=_observations(),
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=ACTION_SCALE, use_default_offset=True
      )
    },
    commands={COMMAND: _command(spec, phase, play)},
    events=_events(spec, phase),
    rewards=build_rewards(spec),
    terminations=_terminations(spec),
    curriculum=_curriculum(spec, phase),
    metrics=_metrics(),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=3.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=None,
      njmax=300,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(timestep=SIM_DT, iterations=10, ls_iterations=20, ccd_iterations=50),
    ),
    decimation=spec.decimation,
    episode_length_s=EPISODE_LENGTH_S,
  )
  assert len(LEG_NAMES) == len(FOOT_SITES)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}  # no SGS: speeds uniform over the range, changed every 4-8 s
  return cfg
