"""Reward FUNCTIONS (building blocks). Which ones a gait uses, and with what weights,
is decided per gait in ../rewards/<gait>.py.

Generic, gait-agnostic terms only; anything that needs the schedule reads the
GaitCommand clock so every term agrees on it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
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


def forward_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_command: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """clamp(v_x / v_cmd, 0, 1): share of the commanded forward speed actually reached
  (1 when the command is below min_command, i.e. nothing to track)."""
  asset: Entity = env.scene[asset_cfg.name]
  v_cmd = env.command_manager.get_command(command_name)[:, 0]
  v = asset.data.root_link_lin_vel_b[:, 0]
  p = (v / v_cmd.clamp(min=min_command)).clamp(0.0, 1.0)
  return torch.where(v_cmd.abs() < min_command, torch.ones_like(p), p)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  xy_weight: float = 0.05,
  gate_by_progress: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-((w_z_cmd - w_z)^2 + xy_weight |w_xy|^2) / std^2), times forward_progress when
  gate_by_progress.

  xy_weight 0.05 is the walking task's; the sharp turn term (track_yaw_fine) uses 0 so
  that it scores the yaw rate alone (with std 0.25 the 0.05 roll/pitch share would weigh
  0.8 w_xy^2, mostly the canter/gallop's natural pitching).

  The gate: in gaits_v4 the second trot fork learned to STAND (v_x -0.05 m/s at every
  command) because standing still scored this term at 0.97 (no yaw rate needed on
  near-straight radii), plus track_yaw_fine and track_turn_lean, while the linear kernel
  gives no gradient 2 m/s away from the command. Gated, these pay nothing at a standstill
  and grow linearly with the speed reached."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  w = asset.data.root_link_ang_vel_b
  err = (cmd[:, 2] - w[:, 2]).square() + xy_weight * w[:, :2].square().sum(dim=1)
  r = torch.exp(-err / std**2)
  return r * forward_progress(env, command_name) if gate_by_progress else r


def body_orientation_roll_pitch(
  env: ManagerBasedRlEnv,
  roll_weight: float = 1.0,
  pitch_weight: float = 1.0,
  command_name: str | None = None,
  gravity: float = 9.81,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """pitch_weight * g_x^2 + roll_weight * (g_y - sin(theta))^2 of the body's projected
  gravity, roll measured from the equilibrium lean of the commanded turn (the walking
  task's term, centaur_rl/tasks/walking/mdp/rewards.py).

  An arc of curvature k = 1 / r at speed v needs a centripetal acceleration v^2 k, which
  leaning into the turn by theta = atan(v^2 k / g) balances; at that lean gravity has
  body-frame y component sin(theta) (positive = leaning left, for a left turn, k > 0).
  v is the measured forward speed, k the command's. Straight (k = 0, ladder / handoff, or
  no command_name) the target is upright, as before.
  """
  g, roll_target = _gravity_and_turn_lean(env, command_name, gravity, asset_cfg)
  return pitch_weight * g[:, 0].square() + roll_weight * (g[:, 1] - roll_target).square()


def _gravity_and_turn_lean(
  env: ManagerBasedRlEnv, command_name: str | None, gravity: float, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """(body-frame gravity [B, 3], target g_y = sin(atan(v^2 k / g)) of the commanded turn [B]).

  g_y > 0 is leaning left, the lean of a left turn (k > 0). v is the measured forward
  speed, k the command's curvature; the target is 0 (upright) when straight or with no
  command_name."""
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    g = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  else:
    g = asset.data.projected_gravity_b
  roll_target = torch.zeros_like(g[:, 1])
  if command_name is not None:
    k = env.command_manager.get_term(command_name).curvature
    v = asset.data.root_link_lin_vel_b[:, 0]
    roll_target = torch.sin(torch.atan(v.square() * k / gravity))
  return g, roll_target


def track_turn_lean(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  gravity: float = 9.81,
  gate_by_progress: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-(g_y - sin(theta))^2 / std^2), times forward_progress when gate_by_progress
  (see track_angular_velocity): reward for leaning into the commanded turn by
  its equilibrium angle theta = atan(v^2 k / g) (the roll target of
  body_orientation_roll_pitch; upright when straight).

  The L2 term alone hardly asks for the lean: it is flat near its minimum, so staying
  upright through an 8.4 deg turn (walk, 1.2 m/s on 1 m) costs only 0.021 * weight per
  step. This kernel scores that 0.01 (std 0.07 ~ 4 deg) against 1.0 when leaning right.
  """
  g, roll_target = _gravity_and_turn_lean(env, command_name, gravity, asset_cfg)
  r = torch.exp(-(g[:, 1] - roll_target).square() / std**2)
  return r * forward_progress(env, command_name) if gate_by_progress else r


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


def angular_momentum_penalty(
  env: ManagerBasedRlEnv, sensor_name: str, command_name: str | None = None
) -> torch.Tensor:
  """|L|^2 of the whole-body angular momentum (subtreeangmom sensor on base_link).

  With command_name, envs on a commanded turn drop the vertical (yaw) component: a yaw
  rate of v / r needs yaw angular momentum, and track_angular_velocity already scores
  the yaw rate. Straight envs are unchanged.
  """
  sensor: BuiltinSensor = env.scene[sensor_name]
  L = sensor.data
  sq = L.square().sum(dim=-1)
  env.extras["log"]["Metrics/angular_momentum_mean"] = sq.sqrt().mean()
  if command_name is not None:
    turning = env.command_manager.get_term(command_name).is_turning()
    sq = torch.where(turning, L[:, :2].square().sum(dim=-1), sq)
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


##
# Gallop terms. Horse references: Witte, Hirst & Wilson 2006 (4389.pdf: footfall timing,
# duty factor, flight and overlap vs speed, 9-17 m/s) and Self Davies, Spence & Wilson
# 2019 (12348_...pdf: force-plate GRFs of galloping Thoroughbreds). See
# docs/multi_policy.md (Gallop) for the numbers. All gait-agnostic in form: they take the
# leg names as parameters and read the GaitCommand clock.
##


def _leg_index(cmd: GaitCommand, leg: str) -> int:
  return next(i for i, g in enumerate(cmd.foot_order) if g.startswith(leg))


def _fore_mask(cmd: GaitCommand, device) -> torch.Tensor:
  return torch.tensor([g.startswith("F") for g in cmd.foot_order], device=device)


def _robot_weight(env: ManagerBasedRlEnv) -> tuple[float, float]:
  """(total mass kg, weight N) of the model (the robot is the only massive body)."""
  m = float(env.sim.mj_model.body_mass.sum())
  return m, m * 9.81


def _grf_substeps(env: ManagerBasedRlEnv, sensor: ContactSensor) -> tuple[torch.Tensor, float]:
  """([B, feet, H, 3] ground reaction force on each foot at every physics substep of the
  last policy step, world frame, z up; substep dt). The sensor's sign convention is
  undone with the sign of z (the ground can only push a foot up)."""
  hist = sensor.data.force_history
  if hist is None:  # sensor without history: one sample held for the whole step
    return _grf_sign(sensor.data.force.unsqueeze(2)), env.step_dt
  return _grf_sign(hist), env.physics_dt


def _grf_sign(f: torch.Tensor) -> torch.Tensor:
  return f * torch.where(f[..., 2:3] < 0, -1.0, 1.0)


def unscheduled_flight(env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
  """1 while every foot is off the ground although the clock schedules at least one in
  stance (negative weight). Galloping horses have exactly one flight per stride (no
  double suspension in Witte 2006's 5642 strides); this blocks the mid-stride
  ("gathered") flight and bounding hops, which `feet_suspension` does not see.
  """
  cmd: GaitCommand = env.command_manager.get_term(command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  airborne = ~(sensor.data.current_contact_time > 0).any(dim=1)
  scheduled = ~cmd.stance_target(lookahead_steps=1).any(dim=1)
  bad = airborne & ~scheduled & cmd.is_moving()
  env.extras["log"]["Metrics/unscheduled_flight"] = bad.float().mean()
  return bad.float()


def flight_duration(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  landing_foot: str = "RL",
  std: float = 0.025,
) -> torch.Tensor:
  """At the touchdown of `landing_foot` that ends a flight: exp(-((t - t_sched) / std)^2),
  t the flight just flown (s), t_sched the clock's flight window (no foot scheduled down).
  Sparse. Unlike `feet_suspension` it also scores flights that are too LONG (hops): the
  horse's flight stays ~27% of the stride at every speed (Witte 2006) and speed comes
  from less overlap between feet, not a longer flight.

  Stateless: at the landing step the other feet are still in the air, so the flight is
  the most recent other lift-off (their smallest air time) minus the time the landing
  foot has already been down.
  """
  cmd: GaitCommand = env.command_manager.get_term(command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  i = _leg_index(cmd, landing_foot)
  others = [j for j in range(cmd.num_feet) if j != i]
  landed = sensor.compute_first_contact(dt=env.step_dt)[:, i]
  contact = sensor.data.current_contact_time > 0
  down_for = sensor.data.current_contact_time[:, i]
  flight = sensor.data.current_air_time[:, others].min(dim=1).values - down_for
  # The landing foot must itself have been up when the last other foot lifted.
  was_up = sensor.data.last_air_time[:, i] >= flight
  hit = landed & ~contact[:, others].any(dim=1) & (flight > 0) & was_up & cmd.is_moving()
  target = cmd.scheduled_flight_fraction() / cmd.stride_freq
  if hit.any():
    env.extras["log"]["Metrics/flight_time_ms"] = 1e3 * flight[hit].mean()
    env.extras["log"]["Metrics/flight_target_ms"] = 1e3 * target[hit].mean()
  return hit.float() * torch.exp(-((flight - target) / std).square())


class footfall_sequence:
  """1 on the touchdown of the LAST foot of `sequence` when the others landed before it,
  in order, within the last stride (positive weight, sparse: once per stride).

  Gallop (right lead): RL, RR, FL, FR = non-lead hind, lead hind, non-lead fore, lead
  fore (Witte 2006, Fig. 1). This separates the transverse gallop from a canter (FL
  before RR) and a rotary gallop (FR before FL). `diagonal_order` cannot be reused: it
  asks the partner foot to still be DOWN, but at the top of the gallop range FL lifts
  before FR lands (the overlap is ~16 ms), which is correct. Strict order: two feet of
  the sequence landing on the same step fail it (a 4-beat gait has four beats).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    cmd: GaitCommand = env.command_manager.get_term(cfg.params["command_name"])
    seq = cfg.params.get("sequence", ("RL", "RR", "FL", "FR"))
    self.idx = [_leg_index(cmd, leg) for leg in seq]
    n, dev = env.num_envs, env.device
    self.t = torch.zeros(n, device=dev)
    self.last_td = torch.full((n, cmd.num_feet), -1e3, device=dev)
    self.prev_end = torch.full((n,), -1e3, device=dev)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.t[env_ids] = 0.0
    self.last_td[env_ids] = -1e3
    self.prev_end[env_ids] = -1e3

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    sequence: tuple[str, ...] = ("RL", "RR", "FL", "FR"),
  ) -> torch.Tensor:
    del sequence  # Resolved in __init__.
    cmd: GaitCommand = env.command_manager.get_term(command_name)
    sensor: ContactSensor = env.scene[sensor_name]
    self.t += env.step_dt
    first = sensor.compute_first_contact(dt=env.step_dt)
    self.last_td = torch.where(first, self.t.unsqueeze(1), self.last_td)
    td = self.last_td[:, self.idx]  # [B, len(sequence)], sequence order
    end = first[:, self.idx[-1]]
    ordered = (td[:, :-1] < td[:, 1:]).all(dim=1)
    fresh = td[:, 0] > self.prev_end  # the first foot landed after the last stride ended
    within = (self.t - td[:, 0]) < 1.0 / cmd.stride_freq
    ok = end & ordered & fresh & within & cmd.is_moving()
    if end.any():
      env.extras["log"]["Metrics/footfall_sequence_ok"] = ok[end].float().mean()
    self.prev_end = torch.where(end, self.t, self.prev_end)
    return ok.float()


class grf_peak_excess:
  """sum over feet of relu(Fz_peak / Fz_pred - margin)^2 (negative weight): vertical
  force far above what the clock's duty factor implies, i.e. stomping.

  Fz_pred = pi * P * m * g / (4 * duty) is the half-sine prediction from duty factor
  (Alexander et al. 1979; Witte 2006 eq. 1), P the share of body weight carried by that
  leg's PAIR. Self Davies 2019 measured it with force plates in galloping horses: the
  prediction is 100% of the recorded peak on average (middle half 90-107%) once P uses
  their measured fore share (~50:50, shifting to the hind legs with speed). Fz_peak is
  the largest of the step's physics substeps. Replaces `soft_landing` for the gallop.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    cmd: GaitCommand = env.command_manager.get_term(cfg.params["command_name"])
    self.fore = _fore_mask(cmd, env.device)
    _, self.weight = _robot_weight(env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    fore_share: tuple[float, float] = (0.632, -0.0195),
    margin: float = 1.3,
  ) -> torch.Tensor:
    cmd: GaitCommand = env.command_manager.get_term(command_name)
    sensor: ContactSensor = env.scene[sensor_name]
    grf, _ = _grf_substeps(env, sensor)
    fz = grf[..., 2].amax(dim=2)  # [B, feet]
    p_fore = (fore_share[0] + fore_share[1] * cmd.speed).clamp(0.3, 0.7).unsqueeze(1)
    share = torch.where(self.fore, p_fore, 1.0 - p_fore)
    pred = math.pi * share * self.weight / (4.0 * cmd.duty.clamp(min=0.05))
    ratio = fz / pred
    loaded = fz > 0.1 * self.weight
    if loaded.any():
      # Mean over loaded foot-steps (not only peaks): a perfect half-sine reads ~0.7.
      env.extras["log"]["Metrics/grf_peak_over_pred"] = ratio[loaded].mean()
      env.extras["log"]["Metrics/grf_peak_bw"] = (fz[loaded] / self.weight).mean()
    excess = (ratio - margin).clamp(min=0.0).square().sum(dim=1)
    return excess * cmd.is_moving().float()


class _StrideImpulse:
  """Per-foot ground-reaction impulse summed over one stride, in the heading frame
  (x forward, z up). A stride runs from one touchdown of `stride_foot` to the next; the
  subclass is called with the finished stride on the step it ends. The first stride after
  a reset is partial and skipped, as is any stride whose length is outside 0.5-1.5 cycles
  (a stumble or a double landing of the stride foot).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    cmd: GaitCommand = env.command_manager.get_term(cfg.params["command_name"])
    self.boundary = _leg_index(cmd, cfg.params.get("stride_foot", "RL"))
    self.fore = _fore_mask(cmd, env.device)
    self.mass, self.weight = _robot_weight(env)
    n, f, dev = env.num_envs, cmd.num_feet, env.device
    self.iz = torch.zeros(n, f, device=dev)
    self.ix = torch.zeros(n, f, device=dev)
    self.t = torch.zeros(n, device=dev)
    self.started = torch.zeros(n, dtype=torch.bool, device=dev)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.iz[env_ids] = 0.0
    self.ix[env_ids] = 0.0
    self.t[env_ids] = 0.0
    self.started[env_ids] = False

  def _step(self, env: ManagerBasedRlEnv, cmd: GaitCommand, sensor: ContactSensor):
    """Accumulate this step; returns (ended [B], iz, ix [B, feet], stride time [B]) where
    iz/ix/T describe the stride that ended this step (garbage where ended is False)."""
    asset: Entity = env.scene["robot"]
    grf, dt_sub = _grf_substeps(env, sensor)
    imp = grf.sum(dim=2) * dt_sub  # [B, feet, 3] world frame
    yaw = yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, imp.shape[1], -1)
    imp = quat_apply_inverse(yaw, imp)
    first = sensor.compute_first_contact(dt=env.step_dt)[:, self.boundary]
    cycles = self.t * cmd.stride_freq
    ended = first & self.started & (cycles > 0.5) & (cycles < 1.5) & cmd.is_moving()
    iz, ix, t = self.iz.clone(), self.ix.clone(), self.t.clone()
    # The stride foot's landing step starts the next stride.
    keep = (~first).float()
    self.iz = self.iz * keep.unsqueeze(1) + imp[..., 2]
    self.ix = self.ix * keep.unsqueeze(1) + imp[..., 0]
    self.t = self.t * keep + env.step_dt
    self.started |= first
    return ended, iz, ix, t


class vertical_impulse_balance(_StrideImpulse):
  """At each stride end: exp(-((s - s_target) / std)^2), s the fore legs' share of the
  stride's vertical impulse (positive weight, sparse).

  Self Davies 2019 (Fig. 3): galloping horses carry ~49:51 fore:hind, not the 57:43 of
  standing and slow gaits, and the share moves to the hind legs with speed: hind % =
  36.8 + 1.215 v_horse. With v_horse = 1.61 v: s_target = 0.632 - 0.0195 v (0.545 at
  4.47 m/s, 0.476 at 8 m/s). An even load spreads the work over all four legs (their
  point: fatigue / injury; here: motor load).
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    stride_foot: str = "RL",
    fore_share: tuple[float, float] = (0.632, -0.0195),
    std: float = 0.05,
  ) -> torch.Tensor:
    del stride_foot  # Resolved in __init__.
    cmd: GaitCommand = env.command_manager.get_term(command_name)
    sensor: ContactSensor = env.scene[sensor_name]
    ended, iz, _, t = self._step(env, cmd, sensor)
    total = iz.sum(dim=1).clamp(min=1e-6)
    share = (iz * self.fore).sum(dim=1) / total
    target = fore_share[0] + fore_share[1] * cmd.speed
    if ended.any():
      env.extras["log"]["Metrics/fore_impulse_share"] = share[ended].mean()
      env.extras["log"]["Metrics/fore_impulse_target"] = target[ended].mean()
      # Sanity: one stride's vertical impulse should be ~ m g T.
      bw = total / (self.weight * t.clamp(min=1e-3))
      env.extras["log"]["Metrics/stride_impulse_bw"] = bw[ended].mean()
    return ended.float() * torch.exp(-((share - target) / std).square())


class propulsion_split(_StrideImpulse):
  """At each stride end: (relu(-Ix_hind) + relu(Ix_fore)) / (m g T) (negative weight,
  sparse): the hind pair's net fore-aft impulse should push forward and the fore pair's
  brake. Self Davies 2019 (Table 1): net cranio-caudal impulse +0.08 / +0.04 N s/kg for
  the non-lead / lead hind, 0 / -0.03 for the non-lead / lead fore. At steady speed the
  two pairs cancel, so this only fixes which pair does what.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    stride_foot: str = "RL",
  ) -> torch.Tensor:
    del stride_foot  # Resolved in __init__.
    cmd: GaitCommand = env.command_manager.get_term(command_name)
    sensor: ContactSensor = env.scene[sensor_name]
    ended, _, ix, t = self._step(env, cmd, sensor)
    hind = (ix * ~self.fore).sum(dim=1)
    fore = (ix * self.fore).sum(dim=1)
    if ended.any():
      env.extras["log"]["Metrics/hind_net_impulse_x"] = (hind[ended] / self.mass).mean()
      env.extras["log"]["Metrics/fore_net_impulse_x"] = (fore[ended] / self.mass).mean()
    cost = (torch.relu(-hind) + torch.relu(fore)) / (self.weight * t.clamp(min=1e-3))
    return ended.float() * cost
