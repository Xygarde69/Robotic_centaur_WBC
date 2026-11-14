"""Gait observations. All read the GaitCommand term (one clock for everything).

Observation layout is identical for every gait (the network's input size can never
change between stages, since each gait starts from the previous gait's weights):
  gait_command   [2]  forward speed command, stride frequency (Hz)
  gait_clock     [8]  sin/cos of each leg's phase
  gait_onehot    [4]  which gait (NUM_GAITS wide, fixed). Constant within a policy, kept
                      so the four policies can later be distilled into one.
  gait_duty      [1]  duty factor of the gait's clock (mean over legs: the gallop's
                      fore and hind duties differ)
Critic only:
  gait_stance    [4]  scheduled stance per foot (next step)
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


def _term(env: ManagerBasedRlEnv, command_name: str) -> GaitCommand:
  return env.command_manager.get_term(command_name)


def gait_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  t = _term(env, command_name)
  return torch.stack([t.speed, t.stride_freq], dim=1)


def gait_clock(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).clock_obs()


def gait_onehot(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).gait_onehot


def gait_duty(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).mean_duty().unsqueeze(1)


def gait_stance(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).stance_target(lookahead_steps=1).float()


def foot_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_air_time is not None
  return sensor.data.current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  f = sensor.data.force.flatten(start_dim=1)
  return torch.sign(f) * torch.log1p(torch.abs(f))
