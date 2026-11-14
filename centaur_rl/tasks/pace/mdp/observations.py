from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .rewards import _command_is_standing, gait_stance_target, resolve_gait_step_table, walk_gait_cycles

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(
    env: ManagerBasedRlEnv,
    period: float,
    command_name: str,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    # Same standing test as the gait reward, so the clock is never hidden while the
    # gait is being rewarded.
    stand_mask = _command_is_standing(env, command_name, command_threshold)
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase


def walk_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Gait clock from the walk command: [sin, cos, first-cycle progress], zero when standing.

    sin/cos are the trot's phase observation on the walk clock. The third entry is
    min(cycles, 1): it tells the startup from rest (legs still waiting for their first
    lift-off) apart from the repeating cycle, which sin/cos alone cannot.
    """
    cycles, walking = walk_gait_cycles(env, command_name, lookahead=False)
    angle = 2.0 * torch.pi * cycles
    obs = torch.stack([torch.sin(angle), torch.cos(angle), cycles.clamp(max=1.0)], dim=1)
    return obs * walking.float().unsqueeze(1)


class gait_target:
    """Scheduled stance (1 = ground, 0 = air) per foot for the next control step.

    Feet are in contact-sensor (model) order.

    The action chosen from this observation is rewarded against the schedule one
    step later, hence the one-step lookahead.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
        self.step_table = resolve_gait_step_table(cfg.params, env)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        phase_durations: list[float],
        stance_table: list[list[int]],
        foot_geoms: tuple[str, ...],
        command_name: str,
        command_threshold: float = 0.1,
    ) -> torch.Tensor:
        del phase_durations, stance_table, foot_geoms  # Baked into step_table.
        return gait_stance_target(
            env, self.step_table, command_name, command_threshold, lookahead_steps=1
        ).float()
