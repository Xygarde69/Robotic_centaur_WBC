"""Per-episode gait metrics (logged, never rewarded): is the policy doing THIS gait?

Reward curves cannot tell a canter from a fast trot that half-satisfies the canter
reward; these can. All per-env (B,), averaged over the episode by the metrics manager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .command import GaitCommand

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.data.current_contact_time > 0


def gait_foot_match(env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
  """Fraction of feet whose contact matches the clock."""
  cmd: GaitCommand = env.command_manager.get_term(command_name)
  return (cmd.stance_target(0) == _contact(env, sensor_name)).float().mean(dim=1)


def gait_exact_match(env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
  """1 when all four feet match the clock."""
  cmd: GaitCommand = env.command_manager.get_term(command_name)
  return (cmd.stance_target(0) == _contact(env, sensor_name)).all(dim=1).float()


def footfall_timing_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Episode-so-far mean worst deviation (cycle fractions) of the footfall spacing from
  the clock's. 0 = perfect. Zero unless the command tracks footfalls."""
  cmd: GaitCommand = env.command_manager.get_term(command_name)
  return cmd.ep_tim_err / cmd.ep_strides.clamp(min=1)


def duty_factor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Measured duty factor (mean share of feet on the ground); compare with the clock's."""
  return _contact(env, sensor_name).float().mean(dim=1)


def flight_phase(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1 while no foot touches the ground (canter / gallop should show this)."""
  return (~_contact(env, sensor_name).any(dim=1)).float()


def forward_speed_error(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  return (asset.data.root_link_lin_vel_b[:, 0] - cmd[:, 0]).abs()


def peak_torque(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Largest |actuator force| this step (episode mean of the per-step peak: torque headroom)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.actuator_force.abs().max(dim=1).values


def mechanical_power(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """sum |tau * qdot| (W); divide by m g v for cost of transport."""
  asset: Entity = env.scene[asset_cfg.name]
  return (asset.data.actuator_force * asset.data.joint_vel).abs().sum(dim=1)
