"""The stage graph: which stages exist, what each starts from, in what order they run.

Per gait g (prev = the slower gait, next = the faster one):

  g-refine   whole-range refinement + turns.  init: fork from prev-refine @ prev.fork_iteration
                                                (walk: from scratch, or --init-checkpoint;
                                                gallop: from scratch, fork_from_previous=False)
                                              banks: prev-up (prev's top-speed states)
  [g-up]     record states of g-refine at g's top speed     (if a faster gait exists)
  [g-down]   record states of g-refine at g's lowest speed  (if a slower gait exists)
  g-handoff  optional fine-tune.        init: continue from g-refine (final)
                                        banks: prev-up and next-down

There is no ladder stage in the plan any more: in run gaits_v2 the walk ladder never
mastered its first speed bin (0% success for 2000 iterations) and the trot forked from
it never tracked speed, while the walk refine did. Each gait now forks from the
previous gait's REFINED policy. (The ladder tasks are still registered, for training a
ladder by hand.)

Handoff stages run last: next-down needs the faster gait's refined policy. The deployed
policy of each gait is g-handoff if enabled, else g-refine. (The neighbour banks used by
g-handoff come from the neighbours' refine policies, not their handoff versions: an
approximation that breaks the refine <-> refine dependency cycle.)

Execution order is gait-major (walk-refine, walk-up, trot-refine, trot-up, ...), which
is a valid topological order and finishes whole gaits early.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from centaur_rl.tasks.Multi_Policy.gaits import GAITS, GaitSpec

PIPELINE_TASK_ID = "Multi-Gait_straight_flat_SGS"
# The ladder's advance rule (env_cfg._sgs: advance_threshold, min_outcomes).
LADDER_ADVANCE_THRESHOLD = 0.8
LADDER_MIN_OUTCOMES = 50
EXPERIMENT_ROOT = "multigait"  # logs/rsl_rl/multigait/<pipeline run>/<stage>/<timestamp>


def stage_task_id(spec: GaitSpec, phase: str) -> str:
  return f"{PIPELINE_TASK_ID}-{spec.title}-{phase.capitalize()}"


def parse_task_id(task_id: str) -> tuple[GaitSpec, str] | None:
  """(gait, phase) of a stage task id; the pipeline task itself is the walk ladder."""
  if task_id == PIPELINE_TASK_ID:
    return GAITS[0], "ladder"
  for g in GAITS:
    for phase in ("ladder", "refine", "handoff"):
      if task_id == stage_task_id(g, phase):
        return g, phase
  return None


def ladder_play_speed(task_id: str, infos: dict | None) -> tuple[float, str] | None:
  """Speed to play a LADDER checkpoint at: the top of the last speed bin it mastered.

  The checkpoint stores the SGS state at save time (rl/runner.py: infos["sgs"]). The
  frontier is the first bin not yet mastered (it advances at >= 80% success), so its
  lower edge = the upper edge of the last mastered bin = the fastest speed the policy
  had learned. With nothing mastered (frontier 0) that is the bottom of the gait's SGS
  range. Returns None for non-ladder checkpoints / tasks.
  """
  parsed = parse_task_id(task_id)
  infos = infos or {}
  sgs = infos.get("sgs")
  stage = (infos.get("stage") or {}).get("name", "")
  is_ladder = stage.endswith("-ladder") or (parsed is not None and parsed[1] == "ladder")
  if parsed is None or sgs is None or not is_ladder:
    return None
  gait = parsed[0]
  ckpt_gait = stage.rsplit("-", 1)[0] if stage else gait.name
  if ckpt_gait != gait.name:
    raise ValueError(f"checkpoint is a {ckpt_gait} ladder but the task is {task_id}")
  lo, hi = gait.sgs_speed_range
  n = gait.speed_bins
  frontier = int(sgs["frontier"])
  edges = [lo + (hi - lo) * i / n for i in range(n + 1)]
  speed = edges[frontier]
  mastered = f"bins 0-{frontier - 1} mastered" if frontier > 0 else "no bin mastered yet"
  # The frontier cannot move past the last bin: check whether that one is mastered too
  # (same rule as the ladder's advance test in env_cfg._sgs: >= 0.8 over >= 50 outcomes).
  if frontier == n - 1 and "hist" in sgs and "hist_count" in sgs:
    count = int(sgs["hist_count"][frontier])
    rate = float(sgs["hist"][frontier].sum()) / max(count, 1)
    if count >= LADDER_MIN_OUTCOMES and rate >= LADDER_ADVANCE_THRESHOLD:
      speed, mastered = edges[n], f"all {n} bins mastered"
  note = (
    f"{gait.name} ladder checkpoint: frontier bin {frontier}/{n - 1} "
    f"({edges[frontier]:.2f}-{edges[frontier + 1]:.2f} m/s), {mastered}"
  )
  return speed, note


@dataclass
class TrainJob:
  name: str  # e.g. "trot-ladder"
  gait: GaitSpec
  phase: str  # ladder | refine | handoff
  iterations: int
  init_mode: str = "none"  # none | fork | continue
  parent: str | None = None  # stage name whose checkpoint this starts from
  init_checkpoint: str | None = None  # explicit checkpoint (first gait only), no parent
  parent_iteration: int | None = None  # None = parent's final checkpoint
  banks: list[str] = field(default_factory=list)  # RecordJob names
  critic_warmup: int = 0
  std_floor: float = 0.0
  normalizer_window: int = 0

  @property
  def task_id(self) -> str:
    return stage_task_id(self.gait, self.phase)

  @property
  def deps(self) -> list[str]:
    return ([self.parent] if self.parent else []) + list(self.banks)


@dataclass
class RecordJob:
  name: str  # e.g. "walk-up"
  gait: GaitSpec
  source: str  # TrainJob whose final policy is recorded
  speed: float  # commanded speed during recording

  @property
  def deps(self) -> list[str]:
    return [self.source]


Job = TrainJob | RecordJob


@dataclass
class PlanOptions:
  gaits: tuple[str, ...] = tuple(g.name for g in GAITS)
  refine_iterations: int | None = None  # None = per-gait budget
  handoff_iterations: int | None = None
  fork_iteration: int | None = None
  critic_warmup: int | None = None
  normalizer_window: int = 2_000_000  # samples; ~20 iterations at 4096 envs x 24 steps
  # Start the FIRST selected gait's refine from this checkpoint instead of from scratch.
  init_checkpoint: str | None = None
  init_mode: str = "continue"  # continue (same gait) | fork (previous gait's policy)
  # Restart named jobs from a checkpoint in continue mode (actor, critic, optimizer),
  # instead of their normal parent: {"trot-refine": "/path/model_700.pt"}.
  stage_init: dict[str, str] = field(default_factory=dict)


def build_plan(opts: PlanOptions | None = None) -> list[Job]:
  opts = opts or PlanOptions()
  gaits = [g for g in GAITS if g.name in opts.gaits]
  if not gaits:
    raise ValueError(f"no gaits selected from {opts.gaits}")
  # A subset must still be a contiguous chain starting anywhere (each forks the previous).
  idx = [GAITS.index(g) for g in gaits]
  if idx != list(range(idx[0], idx[0] + len(idx))):
    raise ValueError(f"gait subset must be contiguous in {[g.name for g in GAITS]}")

  def pick(override, default):
    return default if override is None else override

  jobs: list[Job] = []
  handoffs: list[Job] = []
  for i, g in enumerate(gaits):
    prev = gaits[i - 1] if i > 0 else None
    nxt = gaits[i + 1] if i + 1 < len(gaits) else None
    b = g.budget

    refine = TrainJob(name=f"{g.name}-refine", gait=g, phase="refine",
                      iterations=pick(opts.refine_iterations, b.refine_iterations),
                      banks=[f"{prev.name}-up"] if prev else [])
    if prev is not None and b.fork_from_previous:
      refine.init_mode = "fork"
      refine.parent = f"{prev.name}-refine"
      refine.parent_iteration = pick(opts.fork_iteration, prev.budget.fork_iteration)
    elif opts.init_checkpoint:
      refine.init_mode = opts.init_mode
      refine.init_checkpoint = opts.init_checkpoint
    if refine.init_mode == "fork":
      refine.critic_warmup = pick(opts.critic_warmup, b.critic_warmup_iterations)
      refine.std_floor = b.fork_std_floor
      refine.normalizer_window = opts.normalizer_window
    jobs.append(refine)

    if nxt is not None:
      up = g.v_max if g.up_record_speed is None else g.up_record_speed
      jobs.append(RecordJob(name=f"{g.name}-up", gait=g, source=refine.name, speed=up))
    if prev is not None:
      jobs.append(RecordJob(name=f"{g.name}-down", gait=g, source=refine.name,
                            speed=g.sgs_speed_range[0]))

    h_iters = pick(opts.handoff_iterations, b.handoff_iterations)
    if h_iters > 0 and (prev or nxt):
      handoffs.append(TrainJob(
        name=f"{g.name}-handoff", gait=g, phase="handoff", iterations=h_iters,
        init_mode="continue", parent=refine.name,
        banks=([f"{prev.name}-up"] if prev else []) + ([f"{nxt.name}-down"] if nxt else []),
      ))

  jobs += handoffs
  for name, ckpt in opts.stage_init.items():
    job = next((j for j in jobs if j.name == name), None)
    if not isinstance(job, TrainJob):
      raise ValueError(f"--stage-init {name}: no training job of that name in the plan")
    job.init_mode, job.init_checkpoint = "continue", ckpt
    job.parent, job.parent_iteration = None, None
    job.critic_warmup, job.std_floor, job.normalizer_window = 0, 0.0, 0
  _check_order(jobs)
  return jobs


def final_policies(jobs: list[Job]) -> dict[str, str]:
  """gait -> stage whose final checkpoint is that gait's deployed policy."""
  out: dict[str, str] = {}
  for j in jobs:
    if isinstance(j, TrainJob) and j.phase in ("refine", "handoff"):
      out[j.gait.name] = j.name  # handoff comes after refine, so it wins when present
  return out


def _check_order(jobs: list[Job]) -> None:
  done: set[str] = set()
  for j in jobs:
    missing = [d for d in j.deps if d not in done]
    if missing:
      raise RuntimeError(f"pipeline order bug: {j.name} runs before {missing}")
    done.add(j.name)


def describe(jobs: list[Job]) -> str:
  lines = []
  for k, j in enumerate(jobs):
    if isinstance(j, TrainJob):
      if j.parent is not None:
        src = f"{j.init_mode} <- {j.parent}" + (
          f" @it{j.parent_iteration}" if j.parent_iteration is not None else "")
      elif j.init_checkpoint:
        src = f"{j.init_mode} <- {j.init_checkpoint}"
      else:
        src = "scratch"
      extra = f" warmup={j.critic_warmup} std>={j.std_floor}" if j.init_mode == "fork" else ""
      banks = f" banks={j.banks}" if j.banks else ""
      lines.append(f"{k:2d}. TRAIN  {j.name:15s} {j.iterations:6d} it  {src}{extra}{banks}")
    else:
      lines.append(f"{k:2d}. RECORD {j.name:15s} {j.source} @ {j.speed:.2f} m/s")
  lines.append("deployed: " + ", ".join(f"{g}={s}" for g, s in final_policies(jobs).items()))
  return "\n".join(lines)
