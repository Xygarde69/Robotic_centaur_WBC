"""Run the whole multi-gait pipeline from one command.

    python train.py --task Multi-Gait_straight_flat_SGS [pipeline options] [stage options]

Pipeline options (this module):
  --run-name NAME          pipeline run folder (default: timestamp)
  --resume                 continue a run: skip jobs already marked done in its manifest
  --gaits walk trot ...    contiguous subset of the chain (default: all four)
  --only JOB [JOB ...]     run just these jobs (their dependencies must be done)
  --stop-after JOB         stop after this job
  --refine-iterations N    override every gait's budget (also --handoff-iterations
  --handoff-iterations N     (0 disables handoff stages), --fork-iteration,
  --fork-iteration N         --critic-warmup)
  --init-checkpoint PATH   start the first gait's refine from PATH instead of scratch
  --init-mode MODE           continue (same gait, default) | fork (previous gait)
  --stage-init JOB=PATH    restart JOB from checkpoint PATH in continue mode (actor,
                             critic, optimizer; no warmup / std floor), e.g. to resume
                             a stage after a clock or reward change. Repeatable.
  --record-num-envs N      envs used to record handoff banks (default 512)
  --record-settle-s S      seconds the policy runs before states are recorded (3)
  --record-s S             seconds of recording (6)
  --dry-run                print the plan and exit

Everything else is forwarded unchanged to every training stage's train.py, e.g.
  --env.scene.num-envs 4096 --gpu-ids 0 --agent.logger wandb

Each job runs as its own subprocess (fresh env, fresh CUDA context). Layout:
  logs/rsl_rl/multigait/<run>/
    pipeline.json            manifest: plan, per-job status and results (resume state)
    results/<job>.json       written by each training stage (checkpoints, log dir)
    banks/<job>.pt           handoff banks
    <job>/<timestamp>_<job>/ the stage's normal rsl_rl log dir (model_*.pt, policy.onnx)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from centaur_rl.tasks.Multi_Policy.gaits import GAITS, check_gait_chain
from centaur_rl.tasks.Multi_Policy.gaits import describe as describe_gaits
from centaur_rl.tasks.Multi_Policy.pipeline import (
  EXPERIMENT_ROOT,
  PIPELINE_TASK_ID,
  PlanOptions,
  RecordJob,
  TrainJob,
  build_plan,
  describe,
  final_policies,
  stage_task_id,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def _parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    prog=f"train.py --task {PIPELINE_TASK_ID}",
    description="Multi-gait pipeline (walk -> trot -> canter -> gallop). Unknown options "
    "are forwarded to every training stage.",
  )
  p.add_argument("--run-name", default="")
  p.add_argument("--resume", action="store_true")
  p.add_argument("--gaits", nargs="+", default=[g.name for g in GAITS])
  p.add_argument("--only", nargs="+", default=[])
  p.add_argument("--stop-after", default="")
  p.add_argument("--refine-iterations", type=int)
  p.add_argument("--handoff-iterations", type=int)
  p.add_argument("--fork-iteration", type=int)
  p.add_argument("--critic-warmup", type=int)
  p.add_argument("--normalizer-window", type=int, default=2_000_000)
  p.add_argument("--init-checkpoint", default=None)
  p.add_argument("--init-mode", choices=("continue", "fork"), default="continue")
  p.add_argument("--stage-init", action="append", default=[], metavar="JOB=PATH")
  p.add_argument("--record-num-envs", type=int, default=512)
  p.add_argument("--record-settle-s", type=float, default=3.0)
  p.add_argument("--record-s", type=float, default=6.0)
  p.add_argument("--dry-run", action="store_true")
  return p


def _parse_stage_init(items: list[str]) -> dict[str, str]:
  out = {}
  for item in items:
    name, sep, path = item.partition("=")
    if not sep or not name or not path:
      raise SystemExit(f"--stage-init expects JOB=PATH, got {item!r}")
    if not Path(path).is_file():
      raise SystemExit(f"--stage-init {name}: no such checkpoint {path}")
    out[name] = str(Path(path).resolve())
  return out


class Manifest:
  def __init__(self, path: Path):
    self.path = path
    self.data = json.loads(path.read_text()) if path.exists() else {"jobs": {}}

  def save(self) -> None:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = self.path.with_suffix(".tmp")
    tmp.write_text(json.dumps(self.data, indent=2))
    tmp.replace(self.path)

  def job(self, name: str) -> dict:
    return self.data["jobs"].setdefault(name, {"status": "pending"})

  def done(self, name: str) -> bool:
    return self.job(name).get("status") == "done"


def _pick_checkpoint(result: dict, iteration: int | None) -> str:
  if iteration is None:
    return result["final_checkpoint"]
  ckpts = {int(k): v for k, v in result["checkpoints"].items()}
  ok = [k for k in ckpts if k <= iteration]
  if not ok:
    raise RuntimeError(f"no checkpoint at or before iteration {iteration} in {result['log_dir']}")
  return ckpts[max(ok)]


def _run(cmd: list[str], log_path: Path) -> int:
  print("[pipeline] $ " + " ".join(cmd), flush=True)
  env = dict(os.environ)
  env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with open(log_path, "a") as log:
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:  # tee to terminal and the job log
      sys.stdout.write(line)
      log.write(line)
    return proc.wait()


def _train_cmd(job: TrainJob, run: str, root: Path, manifest: Manifest, forwarded: list[str]) -> list[str]:
  result_file = root / "results" / f"{job.name}.json"
  cmd = [
    sys.executable, str(TRAIN_SCRIPT), "--task", job.task_id,
    "--agent.experiment-name", f"{EXPERIMENT_ROOT}/{run}/{job.name}",
    "--agent.run-name", job.name,
    "--agent.max-iterations", str(job.iterations),
    "--agent.stage.name", job.name,
    "--agent.stage.init-mode", job.init_mode,
    "--agent.stage.result-file", str(result_file),
  ]
  if job.parent:
    ckpt = _pick_checkpoint(manifest.job(job.parent)["result"], job.parent_iteration)
    cmd += ["--agent.stage.init-checkpoint", ckpt]
  elif job.init_checkpoint:
    cmd += ["--agent.stage.init-checkpoint", job.init_checkpoint]
  if job.init_mode == "fork":
    cmd += [
      "--agent.stage.critic-warmup-iterations", str(job.critic_warmup),
      "--agent.stage.std-floor", str(job.std_floor),
      "--agent.stage.normalizer-window", str(job.normalizer_window),
    ]
  if job.banks:
    cmd += ["--agent.stage.handoff-banks", ",".join(manifest.job(b)["result"]["path"] for b in job.banks)]
  return cmd + forwarded


def _record_cmd(job: RecordJob, root: Path, manifest: Manifest, args: argparse.Namespace) -> tuple[list[str], Path]:
  out = root / "banks" / f"{job.name}.pt"
  ckpt = manifest.job(job.source)["result"]["final_checkpoint"]
  cmd = [
    sys.executable, "-m", "centaur_rl.tasks.Multi_Policy.tools.record_handoff",
    "--task", stage_task_id(job.gait, "refine"),
    "--checkpoint", ckpt, "--speed", f"{job.speed:.4f}", "--out", str(out),
    "--num-envs", str(args.record_num_envs),
    "--settle-s", str(args.record_settle_s), "--record-s", str(args.record_s),
  ]
  return cmd, out


def run_pipeline(argv: list[str]) -> int:
  args, forwarded = _parser().parse_known_args(argv)
  opts = PlanOptions(
    gaits=tuple(args.gaits),
    refine_iterations=args.refine_iterations,
    handoff_iterations=args.handoff_iterations,
    fork_iteration=args.fork_iteration,
    critic_warmup=args.critic_warmup,
    normalizer_window=args.normalizer_window,
    init_checkpoint=str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
    init_mode=args.init_mode,
    stage_init=_parse_stage_init(args.stage_init),
  )
  jobs = build_plan(opts)
  print(describe_gaits())
  print(describe(jobs))
  if args.dry_run:
    return 0

  run = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  root = (Path("logs") / "rsl_rl" / EXPERIMENT_ROOT / run).resolve()
  manifest = Manifest(root / "pipeline.json")
  if manifest.data["jobs"] and not args.resume:
    print(f"[pipeline] {root} already has a manifest: pass --resume or a new --run-name")
    return 2
  manifest.data.update({
    "run": run, "task": PIPELINE_TASK_ID, "plan": describe(jobs).splitlines(),
    "gait_warnings": check_gait_chain(), "forwarded_args": forwarded,
  })
  manifest.save()

  names = [j.name for j in jobs]
  for name in args.only:
    if name not in names:
      raise SystemExit(f"--only {name}: not in plan {names}")
  for job in jobs:
    if args.only and job.name not in args.only:
      continue
    if manifest.done(job.name):
      print(f"[pipeline] skip {job.name} (done)")
      continue
    missing = [d for d in job.deps if not manifest.done(d)]
    if missing:
      print(f"[pipeline] cannot run {job.name}: dependencies not done: {missing}")
      return 3

    entry = manifest.job(job.name)
    entry.update({"status": "running", "started": datetime.now().isoformat(timespec="seconds")})
    manifest.save()
    log_path = root / "job_logs" / f"{job.name}.log"
    if isinstance(job, TrainJob):
      rc = _run(_train_cmd(job, run, root, manifest, forwarded), log_path)
      result_file = root / "results" / f"{job.name}.json"
      result = json.loads(result_file.read_text()) if rc == 0 and result_file.exists() else None
    else:
      cmd, out = _record_cmd(job, root, manifest, args)
      rc = _run(cmd, log_path)
      result = {"path": str(out)} if rc == 0 and out.exists() else None

    entry["finished"] = datetime.now().isoformat(timespec="seconds")
    if result is None:
      entry["status"] = "failed"
      entry["returncode"] = rc
      manifest.save()
      print(f"[pipeline] {job.name} FAILED (rc={rc}); see {log_path}. Fix and rerun with "
            f"--run-name {run} --resume")
      return rc or 1
    entry.update({"status": "done", "result": result})
    manifest.save()
    print(f"[pipeline] {job.name} done")
    if args.stop_after == job.name:
      break

  deployed = {
    g: manifest.job(s)["result"]["final_checkpoint"]
    for g, s in final_policies(jobs).items() if manifest.done(s)
  }
  manifest.data["deployed_policies"] = deployed
  manifest.save()
  print("[pipeline] deployed policies:\n" + "\n".join(f"  {g}: {p}" for g, p in deployed.items()))
  return 0


def main_pipeline(argv: list[str]) -> None:
  sys.exit(run_pipeline(argv))
