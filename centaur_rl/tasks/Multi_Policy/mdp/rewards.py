"""Reward FUNCTIONS (building blocks). Which ones a gait uses, and with what weights,
is decided per gait in ../rewards/<gait>.py.

Generic, gait-agnostic terms only; anything that needs the schedule reads the
GaitCommand clock so every term agrees on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .command import GaitCommand

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_forward_velocity(
  env: ManagerBasedRlEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """exp(-(vx - vx_cmd)^2 / std^2). Lateral drift is penalised separately."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  err = asset.data.root_link_lin_vel_b[:, 0] - cmd[:, 0]
  return torch.exp(-err.square() / std**2)


def straight_line(
  env: ManagerBasedRlEnv, std: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """exp(-(vy^2 + wz^2) / std^2): no sideways drift, no yaw (straight-line command)."""
  asset: Entity = env.scene[asset_cfg.name]
  vy = asset.data.root_link_lin_vel_b[:, 1]
  wz = asset.data.root_link_ang_vel_b[:, 2]
  return torch.exp(-(vy.square() + wz.square()) / std**2)


def gait_contact_schedule(
  env: ManagerBasedRlEnv, command_name: str, sensor_name: str, exact_weight: float = 0.0
) -> torch.Tensor:
  """(1 - exact_weight) * fraction of feet matching the clock + exact_weight * [all match].

  Generic clock-matching term (the walking task's feet_gait_clock on the gait clock).
  """
  cmd: GaitCommand = env.command_manager.get_term(command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  contact = sensor.data.current_contact_time > 0
  match = cmd.stance_target(lookahead_steps=1) == contact
  frac = match.float().mean(dim=1)
  exact = match.all(dim=1).float()
  return ((1.0 - exact_weight) * frac + exact_weight * exact) * cmd.is_moving().float()


def feet_contact_in_swing(env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
  """Fraction of feet on the ground while scheduled to swing (negative weight)."""
  cmd: GaitCommand = env.command_manager.get_term(command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  contact = sensor.data.current_contact_time > 0
  return (contact & ~cmd.stance_target(lookahead_steps=1)).float().mean(dim=1)


def touchdown_timing(
  env: ManagerBasedRlEnv, command_name: str, sensor_name: str, std: float = 0.08
) -> torch.Tensor:
  """At each touchdown, exp(-d^2 / std^2) with d the foot's distance (cycle fraction) from
  its scheduled touchdown. Sparse: pays only on the step a foot lands, so a foot that
  lands one step late is barely penalised while the wrong foot landing earns nothing
  (the contact-match terms score both the same). std 0.08 is ~2.5 control steps at 1.5 Hz.
  """
  cmd: GaitCommand = env.command_manager.get_term(command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  landed = sensor.compute_first_contact(dt=env.step_dt).float()
  phase = cmd.leg_phase(lookahead_steps=1)  # 0 = scheduled touchdown, as stance_target
  d = torch.minimum(phase, 1.0 - phase)
  return (landed * torch.exp(-d.square() / std**2)).sum(dim=1) * cmd.is_moving().float()


def feet_suspension(env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
  """1 while the clock schedules a flight phase (no foot in stance) and no foot touches
  the ground. The trot task's feet_suspension on the GaitCommand clock: summed over a
  cycle it is the flight time actually achieved.
  """
  cmd: GaitCommand = env.command_manager.get_term(command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  flight = ~cmd.stance_target(lookahead_steps=1).any(dim=1)
  airborne = ~(sensor.data.current_contact_time > 0).any(dim=1)
  if flight.any():
    env.extras["log"]["Metrics/flight_phase_airborne"] = airborne[flight].float().mean()
  return (flight & airborne & cmd.is_moving()).float()


def feet_excess_contact(
  env: ManagerBasedRlEnv, command_name: str, sensor_name: str, max_feet: int = 2
) -> torch.Tensor:
  """Number of feet on the ground beyond max_feet (negative weight). A trot never has
  3-4 limbs down (Robilliard et al. 2007, Table 1: 0-2); this stops a 4-beat walk-trot
  from collecting most of the per-foot clock match."""
  cmd: GaitCommand = env.command_manager.get_term(command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  down = (sensor.data.current_contact_time > 0).sum(dim=1)
  return (down - max_feet).clamp(min=0).float() * cmd.is_moving().float()


def diagonal_order(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  hind: str = "RR",
  fore: str = "FL",
) -> torch.Tensor:
  """1 on the step the `hind` foot lands while the diagonal `fore` foot is not down yet
  (negative weight). The canter/gallop test of Robilliard et al. 2007 (Table 1): in a
  canter the second limb to land is a forelimb, or a forelimb and hindlimb together; if
  the hind lands first the footfalls are a gallop's. Simultaneous landings are allowed.
  Stateless: it reads the contact sensor, not the clock, so it also catches a hind foot
  that lands on time but with the wrong partner.
  """
  cmd: GaitCommand = env.command_manager.get_term(command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  hi = next(i for i, g in enumerate(cmd.foot_order) if g.startswith(hind))
  fi = next(i for i, g in enumerate(cmd.foot_order) if g.startswith(fore))
  landed = sensor.compute_first_contact(dt=env.step_dt)[:, hi]
  fore_down = sensor.data.current_contact_time[:, fi] > 0
  bad = landed & ~fore_down & cmd.is_moving()
  env.extras["log"]["Metrics/diagonal_order_violations"] = bad.float().mean()
  return bad.float()


def base_roll_pitch_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Squared tilt (projected gravity xy)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, :2].square().sum(dim=1)


def base_roll_pitch_rate_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b[:, :2].square().sum(dim=1)


def base_vertical_velocity_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, 2].square()


def feet_slip(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Sum over feet in contact of the foot's squared horizontal speed."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  contact = (sensor.data.found > 0).float()
  vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2].square().sum(dim=-1)
  return (vel * contact).sum(dim=1)


def torque_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.actuator_force.square().sum(dim=1)


##
# Ported from the walking task (centaur_rl/tasks/walking/mdp/rewards.py). Same maths;
# turning (curvature) and the cycle-time command entry are dropped because the
# multi-gait command is straight-line (lin_x, 0, 0) with the clock in GaitCommand.
##


def _command_speed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """|v_xy| + |w_z| of the command (the walking task's activity measure)."""
  cmd = env.command_manager.get_command(command_name)
  return torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  z_weight: float = 2.0,
  std_rel: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-(|v_xy_cmd - v_xy|^2 + z_weight v_z^2) / s^2), s = max(std, std_rel |v_cmd|).

  z_weight 2 is the walking task's; the trot uses 0.5 (its flight phases need the base
  to bounce). std_rel 0 (the walk) is a fixed kernel width; > 0 widens it with the
  commanded speed so the same relative error scores the same at every speed.
  """
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  v = asset.data.root_link_lin_vel_b
  err = (cmd[:, :2] - v[:, :2]).square().sum(dim=1) + z_weight * v[:, 2].square()
  s = torch.clamp(std_rel * torch.norm(cmd[:, :2], dim=1), min=std)
  return torch.exp(-err / s.square())


def track_angular_velocity(
  env: ManagerBasedRlEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """exp(-((w_z_cmd - w_z)^2 + 0.05 |w_xy|^2) / std^2)."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  w = asset.data.root_link_ang_vel_b
  err = (cmd[:, 2] - w[:, 2]).square() + 0.05 * w[:, :2].square().sum(dim=1)
  return torch.exp(-err / std**2)


def body_orientation_roll_pitch(
  env: ManagerBasedRlEnv,
  roll_weight: float = 1.0,
  pitch_weight: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """pitch_weight * g_x^2 + roll_weight * g_y^2 of the body's projected gravity.

  The walking version measures roll from the lean of the commanded turn; the command
  here is always straight, so the target is upright.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    g = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  else:
    g = asset.data.projected_gravity_b
  return pitch_weight * g[:, 0].square() + roll_weight * g[:, 1].square()


def body_angular_velocity_roll_pitch(
  env: ManagerBasedRlEnv,
  roll_weight: float = 1.0,
  pitch_weight: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Weighted body-frame roll-rate^2 + pitch-rate^2."""
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  w_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  w = quat_apply_inverse(quat, w_w)
  return roll_weight * w[:, 0].square() + pitch_weight * w[:, 1].square()


def angular_momentum_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """|L|^2 of the whole-body angular momentum (subtreeangmom sensor on base_link)."""
  sensor: BuiltinSensor = env.scene[sensor_name]
  sq = sensor.data.square().sum(dim=-1)
  env.extras["log"]["Metrics/angular_momentum_mean"] = sq.sqrt().mean()
  return sq


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  speed_normalized: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """sum over feet of |z_foot - target| * |v_xy_foot| (moving feet should be at target).

  speed_normalized divides by the commanded speed |cmd_xy|: swing-foot speed scales with
  body speed, so the same height error then costs the same at every speed (the weight is
  per m/s of command).
  """
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  vel = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  cost = ((z - target_height).abs() * vel).sum(dim=1)
  if command_name is not None:
    cost = cost * (_command_speed(env, command_name) > command_threshold).float()
    if speed_normalized:
      cmd = env.command_manager.get_command(command_name)
      cost = cost / torch.norm(cmd[:, :2], dim=1).clamp(min=command_threshold)
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Sum of contact-force magnitudes of the feet touching down this step."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = torch.norm(sensor.data.force, dim=-1)
  first = sensor.compute_first_contact(dt=env.step_dt).float()
  impact = force * first
  env.extras["log"]["Metrics/landing_force_mean"] = impact.sum() / first.sum().clamp(min=1)
  cost = impact.sum(dim=1)
  if command_name is not None:
    cost = cost * (_command_speed(env, command_name) > command_threshold).float()
  return cost


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """sum (q - q_default)^2 while the command is a stand."""
  asset: Entity = env.scene[asset_cfg.name]
  diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  standing = (_command_speed(env, command_name) <= command_threshold).float()
  return diff.square().sum(dim=1) * standing


class variable_posture:
  """exp(-mean((q - q_default)^2 / std^2)) with per-joint stds by command speed.

  std_standing below walking_threshold, std_walking up to running_threshold,
  std_running above. Stds are {joint-name regex: std}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.default_joint_pos = asset.data.default_joint_pos
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    def stds(key: str) -> torch.Tensor:
      _, _, values = resolve_matching_names_values(data=cfg.params[key], list_of_strings=joint_names)
      return torch.tensor(values, device=env.device, dtype=torch.float32)

    self.std_standing = stds("std_standing")
    self.std_walking = stds("std_walking")
    self.std_running = stds("std_running")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    speed = _command_speed(env, command_name).unsqueeze(1)
    std = torch.where(
      speed < walking_threshold,
      self.std_standing,
      torch.where(speed < running_threshold, self.std_walking, self.std_running),
    )
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - self.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-(err.square() / std.square()).mean(dim=1))
