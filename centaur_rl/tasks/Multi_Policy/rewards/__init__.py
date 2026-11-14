"""Per-gait reward sets. One module per gait, each exposing `rewards(spec) -> dict`.

The env config calls `build_rewards(spec)`, so a gait's reward design is entirely inside
its own module and changing one gait never touches another.
"""

from __future__ import annotations

from collections.abc import Callable

from mjlab.managers.reward_manager import RewardTermCfg

from centaur_rl.tasks.Multi_Policy.gaits import GaitSpec

from . import canter, gallop, trot, walk

REWARD_BUILDERS: dict[str, Callable[[GaitSpec], dict[str, RewardTermCfg]]] = {
  "walk": walk.rewards,
  "trot": trot.rewards,
  "canter": canter.rewards,
  "gallop": gallop.rewards,
}


def build_rewards(spec: GaitSpec) -> dict[str, RewardTermCfg]:
  if spec.name not in REWARD_BUILDERS:
    raise KeyError(f"no reward module for gait '{spec.name}' (add rewards/{spec.name}.py)")
  return REWARD_BUILDERS[spec.name](spec)
