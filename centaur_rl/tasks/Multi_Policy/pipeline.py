"""The stage graph: which stages exist, what each starts from, in what order they run.

Per gait g (prev = the slower gait, next = the faster one):

  g-ladder   SGS speed ladder.          init: fork from prev-ladder @ prev.fork_iteration
                                          (walk: from scratch)
  g-refine   whole-range refinement.    init: continue from g-ladder (final)
                                        banks: prev-up (prev's top-speed states)
  [g-up]     record states of g-refine at g's top speed     (if a faster gait exists)
  [g-down]   record states of g-refine at g's lowest speed  (if a slower gait exists)
  g-handoff  optional fine-tune.        init: continue from g-refine (final)
                                        banks: prev-up and next-down

The forks chain through the LADDERS (trot starts from the walk ladder's checkpoint, not
the refined walk), so a gait does not wait for the previous gait's 10k-iteration
refinement, and the forked policy still has its exploration (std / entropy) left.
Refine stages need prev-up, i.e. prev's refined policy, so they still run in order.

Handoff stages run last: next-down needs the faster gait's refined policy. The deployed
policy of each gait is g-handoff if enabled, else g-refine. (The neighbour banks used by
g-handoff come from the neighbours' refine policies, not their handoff versions: an
approximation that breaks the refine <-> refine dependency cycle.)

Execution order is gait-major (walk-ladder, walk-refine, walk-up, trot-ladder, ...),
which is a valid topological order and finishes whole gaits early.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from centaur_rl.tasks.Multi_Policy.gaits import GAITS, GaitSpec

PIPELINE_TASK_ID = "Multi-Gait_straight_flat_SGS"
EXPERIMENT_ROOT = "multigait"  # logs/rsl_rl/multigait/<pipeline run>/<stage>/<timestamp>


def stage_task_id(spec: GaitSpec, phase: str) -> str:
  return f"{PIPELINE_TASK_ID}-{spec.title}-{phase.capitalize()}"


@dataclass
class TrainJob:
  name: str  # e.g. "trot-ladder"
  gait: GaitSpec
  phase: str  # ladder | refine | handoff
  iterations: int
  init_mode: str = "none"  # none | fork | continue
  parent: str | None = None  # stage name whose checkpoint this starts from
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
  ladder_iterations: int | None = None  # None = per-gait budget
  refine_iterations: int | None = None
  handoff_iterations: int | None = None
  fork_iteration: int | None = None
  critic_warmup: int | None = None
  normalizer_window: int = 2_000_000  # samples; ~20 iterations at 4096 envs x 24 steps


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

    ladder = TrainJob(name=f"{g.name}-ladder", gait=g, phase="ladder",
                      iterations=pick(opts.ladder_iterations, b.ladder_iterations))
    if prev is not None:
      ladder.init_mode = "fork"
      ladder.parent = f"{prev.name}-ladder"
      ladder.parent_iteration = pick(opts.fork_iteration, prev.budget.fork_iteration)
      ladder.critic_warmup = pick(opts.critic_warmup, b.critic_warmup_iterations)
      ladder.std_floor = b.fork_std_floor
      ladder.normalizer_window = opts.normalizer_window
    jobs.append(ladder)

    refine = TrainJob(name=f"{g.name}-refine", gait=g, phase="refine",
                      iterations=pick(opts.refine_iterations, b.refine_iterations),
                      init_mode="continue", parent=ladder.name,
                      banks=[f"{prev.name}-up"] if prev else [])
    jobs.append(refine)

    if nxt is not None:
      jobs.append(RecordJob(name=f"{g.name}-up", gait=g, source=refine.name, speed=g.v_max))
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
      src = "scratch" if j.parent is None else (
        f"{j.init_mode} <- {j.parent}" + (f" @it{j.parent_iteration}" if j.parent_iteration is not None else "")
      )
      extra = f" warmup={j.critic_warmup} std>={j.std_floor}" if j.init_mode == "fork" else ""
      banks = f" banks={j.banks}" if j.banks else ""
      lines.append(f"{k:2d}. TRAIN  {j.name:15s} {j.iterations:6d} it  {src}{extra}{banks}")
    else:
      lines.append(f"{k:2d}. RECORD {j.name:15s} {j.source} @ {j.speed:.2f} m/s")
  lines.append("deployed: " + ", ".join(f"{g}={s}" for g, s in final_policies(jobs).items()))
  return "\n".join(lines)
