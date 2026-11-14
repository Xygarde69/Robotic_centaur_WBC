"""Register one task per (gait, phase) stage plus the pipeline task itself.

  Multi-Gait_straight_flat_SGS                 the whole pipeline (train.py hands it to
                                               orchestrator.py). Its env/agent configs
                                               are the walk ladder's, so play/list work.
  Multi-Gait_straight_flat_SGS-<Gait>-<Phase>  one stage, trainable on its own, e.g.
                                               ...-Trot-Ladder, ...-Gallop-Refine.
"""

from mjlab.tasks.registry import register_mjlab_task

from centaur_rl.tasks.Multi_Policy.env_cfg import PHASES, make_env_cfg
from centaur_rl.tasks.Multi_Policy.gaits import GAITS, WALK
from centaur_rl.tasks.Multi_Policy.pipeline import EXPERIMENT_ROOT, PIPELINE_TASK_ID, stage_task_id
from centaur_rl.tasks.Multi_Policy.rl import MultiGaitRunner, multigait_ppo_cfg

_ITERS = {
  "ladder": lambda g: g.budget.ladder_iterations,
  "refine": lambda g: g.budget.refine_iterations,
  "handoff": lambda g: max(g.budget.handoff_iterations, 1),
}

for _gait in GAITS:
  for _phase in PHASES:
    register_mjlab_task(
      task_id=stage_task_id(_gait, _phase),
      env_cfg=make_env_cfg(_gait, _phase),
      play_env_cfg=make_env_cfg(_gait, _phase, play=True),
      rl_cfg=multigait_ppo_cfg(
        f"{EXPERIMENT_ROOT}/{_gait.name}-{_phase}", _ITERS[_phase](_gait), _gait.control_dt
      ),
      runner_cls=MultiGaitRunner,
    )

register_mjlab_task(
  task_id=PIPELINE_TASK_ID,
  env_cfg=make_env_cfg(WALK, "ladder"),
  play_env_cfg=make_env_cfg(WALK, "ladder", play=True),
  rl_cfg=multigait_ppo_cfg(f"{EXPERIMENT_ROOT}/walk-ladder", WALK.budget.ladder_iterations),
  runner_cls=MultiGaitRunner,
)
