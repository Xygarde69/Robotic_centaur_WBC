from mjlab.tasks.registry import register_mjlab_task
from centaur_rl.tasks.pace.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  centaur_flat_env_cfg,
  centaur_rough_env_cfg,
  centaur_rough_sgs_env_cfg,
)
from .rl_cfg import centaur_ppo_runner_cfg

register_mjlab_task(
  task_id="Centaur-Pace-Rough",
  env_cfg=centaur_rough_env_cfg(),
  play_env_cfg=centaur_rough_env_cfg(play=True),
  rl_cfg=centaur_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Centaur-Pace-Flat",
  env_cfg=centaur_flat_env_cfg(),
  play_env_cfg=centaur_flat_env_cfg(play=True),
  rl_cfg=centaur_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Same tasks trained with Success-Guided Sampling (arXiv:2610.12465) instead of curricula.
register_mjlab_task(
  task_id="Centaur-Pace-Rough-SGS",
  env_cfg=centaur_rough_sgs_env_cfg(),
  play_env_cfg=centaur_rough_sgs_env_cfg(play=True),
  rl_cfg=centaur_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
