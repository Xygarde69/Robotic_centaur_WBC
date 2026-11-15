"""Drive all four gait policies (walk, trot, canter, gallop) with a DualSense.

    python Multiplay.py                      # bundled policies (policies/multigait/)
    python Multiplay.py --run gaits_v4       # policies of a pipeline run in logs/
    python Multiplay.py --gallop <model_N.pt>   # override one gait

Controls (DualSense, Linux joystick API):
  R2 (hold)   speed up    (rate scales with how far the trigger is pressed)
  L2 (hold)   slow down
  Triangle    next gait up    (walk -> trot -> canter -> gallop)
  Circle      next gait down
  L1 / R1     turn left / right (the turn tightens while held, straightens on release)

How it works:
  * One simulation for every gait, at the gallop's 100 Hz control rate (physics 5 ms,
    2 substeps per step). The walk, trot and canter policies were trained at 50 Hz: they
    are queried every second step and their action is held for both, which is exactly
    what they saw in training (decimation 4).
  * The gait command (clock, stride frequency, duty, gait one-hot, speed range, turn
    limits) is switched live to the selected gait; the stride phase carries on.
  * The commanded speed stays inside the selected gait's trained range. Switching gait
    clamps it into the new gait's range (a step change, as refine trained with its
    mid-episode speed resamples). Switch inside the overlaps for a smooth change:
    walk/trot 1.20-1.30, trot/canter 2.24-2.80, canter/gallop 4.47-6.00 m/s.
  * Turns are arcs: curvature = stick * min(1 / r_min, 0.4 g / v^2) for the gait,
    yaw rate = v * curvature, as in training.
  * After a fall the robot respawns at rest in the walk.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from dataclasses import asdict  # noqa: E402

import torch  # noqa: E402

import mjlab.tasks  # noqa: E402, F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

import centaur_rl.tasks  # noqa: E402, F401
from centaur_rl.tasks.Multi_Policy.gaits import GAITS, SIM_DT, GaitSpec  # noqa: E402
from centaur_rl.tasks.Multi_Policy.pipeline import EXPERIMENT_ROOT, stage_task_id  # noqa: E402
from centaur_rl.tasks.Multi_Policy.tools import dualsense as ds  # noqa: E402

CONTROL_DECIMATION = 2  # 100 Hz, the gallop's rate (the fastest of the four)
BUNDLED = REPO_ROOT / "policies" / "multigait"  # <gait>.pt, committed with the repo


# ---------------------------------------------------------------------- checkpoints
def latest_checkpoint(stage_dir: Path) -> Path | None:
  runs = sorted(p for p in stage_dir.glob("*/") if p.is_dir())
  for run in reversed(runs):
    ckpts = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if ckpts:
      return ckpts[-1]
  return None


def resolve_checkpoints(run_dir: Path | None, overrides: dict[str, str | None]) -> dict[str, Path]:
  """Per gait: --<gait> override; else, with --run, the run's refine result (pipeline.json)
  or the newest checkpoint of a stage still training; else the bundled policies/multigait/."""
  if run_dir is None:
    out = {}
    for g in GAITS:
      path = Path(overrides[g.name]).resolve() if overrides.get(g.name) else BUNDLED / f"{g.name}.pt"
      if not path.exists():
        raise SystemExit(f"no {g.name} policy at {path} (pass --{g.name} <model.pt> or --run <name>)")
      out[g.name] = path
    return out
  manifest = run_dir / "pipeline.json"
  jobs = json.loads(manifest.read_text())["jobs"] if manifest.exists() else {}
  out = {}
  for g in GAITS:
    if overrides.get(g.name):
      out[g.name] = Path(overrides[g.name]).resolve()
      continue
    job = jobs.get(f"{g.name}-refine", {})
    final = (job.get("result") or {}).get("final_checkpoint") if job.get("status") == "done" else None
    path = Path(final) if final else latest_checkpoint(run_dir / f"{g.name}-refine")
    if path is None or not path.exists():
      raise SystemExit(f"no {g.name} policy in {run_dir} (pass --{g.name} <model.pt>)")
    out[g.name] = path
  return out


# ---------------------------------------------------------------------- gait switching
def set_gait(cmd, spec: GaitSpec) -> None:
  """Switch the live GaitCommand to `spec`'s clock, speed range and turn limits."""
  c = cmd.cfg
  c.gait_name, c.gait_index = spec.name, spec.index
  c.speed_range = spec.sgs_speed_range
  c.offsets, c.duty_factor = spec.clock.offsets, spec.clock.duty_factor
  c.air_time, c.duty_min, c.duty_max = spec.clock.air_time, spec.clock.duty_min, spec.clock.duty_max
  c.stance_sweep, c.f_min, c.f_max = spec.stride.stance_sweep, spec.stride.f_min, spec.stride.f_max
  c.f_slope, c.f_offset = spec.stride.slope, spec.stride.offset
  c.turn_radii, c.turn_max_lat_accel = spec.turn_radii, spec.turn_max_lat_accel
  c.track_footfalls = False  # metric only, built for the first gait's clock
  perm = [list(c.foot_geoms).index(name) for name in cmd.foot_order]
  dev = cmd.device
  cmd.offsets = torch.tensor(c.offsets, device=dev)[perm].view(1, -1)
  if c.air_time is None:
    cmd._air_time = None
  else:
    air = c.air_time if isinstance(c.air_time, tuple) else (c.air_time,) * cmd.num_feet
    cmd._air_time = torch.tensor(air, device=dev)[perm].view(1, -1)
  cmd.gait_onehot.zero_()
  cmd.gait_onehot[:, spec.index] = 1.0
  cmd.from_rest[:] = False  # the startup hold is for a standing start only


# ---------------------------------------------------------------------- controller
class Controller:
  """Installed as GaitCommand.joystick: called every control step with (command, dt)."""

  def __init__(self, pad, accel: float, decel: float, ramp_time: float = 1.5, release_time: float = 0.5):
    self.pad = pad
    self.accel, self.decel = accel, decel
    self.ramp_time, self.release_time = ramp_time, release_time
    self.gait = 0
    self.v = GAITS[0].sgs_speed_range[0]
    self.s = 0.0  # turn stick in [-1, 1], + = left
    self.switched = False
    self._since_print = 0.0
    self._pending_reset = False

  @property
  def spec(self) -> GaitSpec:
    return GAITS[self.gait]

  def _switch(self, cmd, k: int) -> None:
    k = max(0, min(len(GAITS) - 1, k))
    if k == self.gait:
      return
    old = self.spec
    self.gait = k
    set_gait(cmd, self.spec)
    lo, hi = self.spec.sgs_speed_range
    v0 = self.v
    self.v = min(max(self.v, lo), hi)
    jump = "" if abs(self.v - v0) < 1e-6 else f" (speed {v0:.2f} -> {self.v:.2f} m/s, outside the overlap)"
    print(f"\n[gait] {old.name} -> {self.spec.name}{jump}")
    self.switched = True

  def respawn(self, cmd) -> None:
    """After a fall: back to the walk at its lowest speed, straight."""
    self.gait = 0
    set_gait(cmd, self.spec)
    cmd.from_rest[:] = True
    self.v, self.s = self.spec.sgs_speed_range[0], 0.0
    self.switched = True
    print("\n[reset] fell: respawned at rest in the walk")

  def __call__(self, cmd, dt: float) -> None:
    pad = self.pad
    pad.poll()
    up, down = pad.presses(ds.BTN_NORTH), pad.presses(ds.BTN_EAST)
    if up or down:
      self._switch(cmd, self.gait + up - down)
    lo, hi = self.spec.sgs_speed_range
    self.v += (self.accel * pad.trigger(ds.ABS_RZ) - self.decel * pad.trigger(ds.ABS_Z)) * dt
    self.v = min(max(self.v, lo), hi)
    left, right = pad.pressed(ds.BTN_TL), pad.pressed(ds.BTN_TR)
    if left != right:
      self.s += (1.0 if left else -1.0) * dt / self.ramp_time
    elif not left:
      step = dt / self.release_time
      self.s = max(0.0, self.s - step) if self.s > 0 else min(0.0, self.s + step)
    self.s = max(-1.0, min(1.0, self.s))
    ids = torch.arange(cmd.num_envs, device=cmd.device)
    cmd._set_speed(ids, torch.full((cmd.num_envs,), self.v, device=cmd.device))
    cmd.set_stick_turn(self.s)
    self._since_print += dt
    if self._since_print >= 0.2:
      self._since_print = 0.0
      self._hud(cmd)

  def _hud(self, cmd) -> None:
    lo, hi = self.spec.sgs_speed_range
    v_meas = float(cmd.robot.data.root_link_lin_vel_b[0, 0])
    w = float(cmd.yaw_rate[0])
    turn = "straight" if abs(w) < 1e-3 else f"{'L' if w > 0 else 'R'} r {self.v / abs(w):5.1f} m"
    hint = []
    if self.gait + 1 < len(GAITS) and GAITS[self.gait + 1].sgs_speed_range[0] <= self.v:
      hint.append(f"Triangle -> {GAITS[self.gait + 1].name}")
    if self.gait > 0 and self.v <= GAITS[self.gait - 1].sgs_speed_range[1]:
      hint.append(f"Circle -> {GAITS[self.gait - 1].name}")
    print(
      f"\r[{self.spec.name:>6}] cmd {self.v:4.2f} m/s (actual {v_meas:4.2f}) range {lo:.2f}-{hi:.2f} | "
      f"{turn:<12} | {' / '.join(hint) or 'no clean shift here':<34}",
      end="", flush=True,
    )


# ---------------------------------------------------------------------- policy switch
class MultiGaitPolicy:
  """policy(obs) for the viewer, called once per 100 Hz step. Queries the selected gait's
  policy at its own control rate and holds its action in between."""

  def __init__(self, env, ctrl: Controller, policies: dict):
    self.env, self.ctrl, self.policies = env, ctrl, policies
    self.cmd = env.unwrapped.command_manager.get_term("gait")
    self.action = None
    self.since = 0
    self.started = False

  def __call__(self, obs):
    unwrapped = self.env.unwrapped
    if self.started and int(unwrapped.episode_length_buf[0]) == 0:
      self.ctrl.respawn(self.cmd)
    self.started = True
    spec = self.ctrl.spec
    every = spec.decimation // CONTROL_DECIMATION  # 2 for the 50 Hz gaits, 1 for the gallop
    if self.action is None or self.ctrl.switched or self.since >= every:
      with torch.no_grad():
        self.action = self.policies[spec.name](obs)
      self.since = 0
      self.ctrl.switched = False
    self.since += 1
    return self.action


# ---------------------------------------------------------------------- main
def build(args):
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  run_dir = (Path(__file__).resolve().parent / "logs" / "rsl_rl" / EXPERIMENT_ROOT / args.run) if args.run else None
  ckpts = resolve_checkpoints(run_dir, {g.name: getattr(args, g.name) for g in GAITS})
  for name, path in ckpts.items():
    print(f"[INFO]: {name:>6}: {path}")

  first = GAITS[0]
  env_cfg = load_env_cfg(stage_task_id(first, "refine"), play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.decimation = CONTROL_DECIMATION
  for sensor in env_cfg.scene.sensors:
    if hasattr(sensor, "history_length"):
      sensor.history_length = CONTROL_DECIMATION
  assert abs(env_cfg.sim.mujoco.timestep - SIM_DT) < 1e-9
  for g in GAITS:
    if g.decimation % CONTROL_DECIMATION:
      raise SystemExit(f"{g.name} decimation {g.decimation} is not a multiple of {CONTROL_DECIMATION}")
  gait_cmd = env_cfg.commands["gait"]
  gait_cmd.resample_mid_episode = False
  gait_cmd.resampling_time_range = (1e6, 1e6)
  gait_cmd.debug_vis = True
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  venv = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(stage_task_id(first, "refine")).clip_actions)

  policies = {}
  for g in GAITS:
    task = stage_task_id(g, "refine")
    agent_cfg = load_rl_cfg(task)
    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device=device)
    runner.load(str(ckpts[g.name]), load_cfg={"actor": True}, strict=True, map_location=device)
    policies[g.name] = runner.get_inference_policy(device=device)
  return env, venv, policies


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--run", default=None, help="pipeline run under scripts/logs/rsl_rl/multigait/ "
                 "(default: the bundled policies in policies/multigait/)")
  for g in GAITS:
    p.add_argument(f"--{g.name}", default=None, help=f"{g.name} checkpoint (default: from the run)")
  p.add_argument("--accel", type=float, default=1.0, help="m/s^2 at full R2")
  p.add_argument("--decel", type=float, default=1.5, help="m/s^2 at full L2")
  p.add_argument("--viewer", choices=("native", "viser"), default="native")
  p.add_argument("--device", default=None)
  args = p.parse_args()

  pad = ds.Gamepad(required_buttons=(ds.BTN_TL, ds.BTN_TR, ds.BTN_NORTH, ds.BTN_EAST), required_axes=(ds.ABS_Z, ds.ABS_RZ))
  print(f"[INFO]: Gamepad {pad.name} ({pad.path})")
  print("[INFO]: R2 faster, L2 slower, Triangle gait up, Circle gait down, L1 / R1 turn")

  env, venv, policies = build(args)
  cmd = env.command_manager.get_term("gait")
  ctrl = Controller(pad, args.accel, args.decel)
  set_gait(cmd, ctrl.spec)
  cmd.from_rest[:] = True
  cmd.joystick = ctrl
  policy = MultiGaitPolicy(venv, ctrl, policies)

  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  try:
    (NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer)(venv, policy).run()
  finally:
    print()
    pad.close()
    env.close()


if __name__ == "__main__":
  main()
