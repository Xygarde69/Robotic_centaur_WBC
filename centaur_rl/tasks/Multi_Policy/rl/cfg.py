"""PPO config + the per-stage runtime knobs the pipeline sets for each stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class MultiGaitStageCfg:
  """How a stage starts. Set by the orchestrator, overridable as --agent.stage.<field>."""

  name: str = ""
  init_mode: Literal["none", "fork", "continue"] = "none"
  """none: from scratch. fork: actor only from init_checkpoint (new gait: fresh critic,
  fresh optimizer, critic warmup, std floor, normalizer re-adaptation). continue: actor,
  critic and optimizer (same gait, next phase)."""
  init_checkpoint: str = ""
  critic_warmup_iterations: int = 0
  """fork: first N iterations update the critic only (actor frozen), so the new gait's
  advantages are not computed from a random value function."""
  std_floor: float = 0.0
  """fork: raise the action std to at least this (the previous gait shrank it)."""
  normalizer_window: int = 0
  """fork: cap the actor obs-normalizer sample count to this, so its running mean/std
  re-adapt to the new gait's observation ranges (it is a cumulative average: after a
  full gait's training new samples would barely move it). 0 = leave as loaded."""
  handoff_banks: str = ""
  """Comma-separated handoff state banks (tools/record_handoff.py) to start episodes from."""
  handoff_prob: float = -1.0
  """Share of resets from the banks; < 0 = the phase default from the env config."""
  result_file: str = ""
  """JSON written at the end (final checkpoint, log dir): how the orchestrator chains."""


@dataclass
class MultiGaitRunnerCfg(RslRlOnPolicyRunnerCfg):
  stage: MultiGaitStageCfg = field(default_factory=MultiGaitStageCfg)


# Values for a 50 Hz policy. Other control rates keep the same horizons in SECONDS:
# gamma 0.99 = a 2 s discount horizon, lambda 0.95 = 0.4 s, 24 steps = 0.48 s rollouts.
REFERENCE_DT = 0.02
GAMMA = 0.99
LAM = 0.95
NUM_STEPS_PER_ENV = 24


def multigait_ppo_cfg(
  experiment_name: str, max_iterations: int, control_dt: float = REFERENCE_DT
) -> MultiGaitRunnerCfg:
  k = control_dt / REFERENCE_DT  # 0.5 at 100 Hz
  return MultiGaitRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=GAMMA**k,  # 0.995 at 100 Hz
      lam=LAM**k,  # 0.975 at 100 Hz
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=round(NUM_STEPS_PER_ENV / k),  # 48 at 100 Hz
    max_iterations=max_iterations,
  )
