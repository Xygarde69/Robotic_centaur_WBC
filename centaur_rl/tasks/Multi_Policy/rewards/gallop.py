"""Gallop: 4-beat transverse (RL, RR, FL, FR), long flight, large pitch oscillation.

PLACEHOLDER: the common terms with the generic clock reward. Design this gait's own
terms here (e.g. foot clearance, flight-phase timing, pitch tolerance, lead-leg
symmetry) and return them from `rewards`.
"""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg

from centaur_rl.tasks.Multi_Policy.gaits import GALLOP, GaitSpec

from .common import common_rewards


def rewards(spec: GaitSpec = GALLOP) -> dict[str, RewardTermCfg]:
  terms = common_rewards(spec)
  # TODO(gallop): gait-specific terms and weights.
  return terms
