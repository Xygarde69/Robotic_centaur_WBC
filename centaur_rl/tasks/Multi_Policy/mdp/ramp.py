"""Penalty ramp: scale chosen reward weights up from a fraction to their full value.

For a stage trained from scratch (the gallop): a random policy pays far more in
regularisation (action rate, joint acceleration, foot slip, torque) per second alive than
the one-off termination penalty, so PPO learns to fall at once. In gaits_v4 a random
gallop policy netted about -15 per second alive against -4 for terminating. Starting
those terms at `start_scale` and ramping them linearly to full weight over `ramp_steps`
env steps (= iterations x steps per env) lets it learn to stay up and move first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class PenaltyRamp:
  """Curriculum term (called at resets): sets weight = base * scale(common_step_counter)."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self.terms: tuple[str, ...] = tuple(p["terms"])
    self.start_scale = float(p.get("start_scale", 0.1))
    self.ramp_steps = max(int(p["ramp_steps"]), 1)
    rm = env.reward_manager
    self.base = {name: rm.get_term_cfg(name).weight for name in self.terms}
    self._apply(env, self.start_scale)
    print(f"[PenaltyRamp] {self.terms} from x{self.start_scale} to x1 over {self.ramp_steps} env steps")

  def _apply(self, env: ManagerBasedRlEnv, scale: float) -> None:
    rm = env.reward_manager
    for name, w in self.base.items():
      rm.get_term_cfg(name).weight = w * scale

  def scale_at(self, step: int) -> float:
    frac = min(step / self.ramp_steps, 1.0)
    return self.start_scale + (1.0 - self.start_scale) * frac

  def __call__(self, env: ManagerBasedRlEnv, env_ids, **_params) -> dict[str, torch.Tensor]:
    scale = self.scale_at(env.common_step_counter)
    self._apply(env, scale)
    return {"scale": torch.tensor(scale)}

  def reset(self, env_ids=None) -> None:
    pass
