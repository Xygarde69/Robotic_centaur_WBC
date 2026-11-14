from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def base_too_low(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=()),
) -> torch.Tensor:
  """Base z minus mean foot-site z below minimum_height (only the feet collide)."""
  asset: Entity = env.scene[asset_cfg.name]
  base_z = asset.data.root_link_pos_w[:, 2]
  feet_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2].mean(dim=-1)
  return (base_z - feet_z) < minimum_height
