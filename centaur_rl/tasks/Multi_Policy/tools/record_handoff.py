"""Record a handoff bank: states of a trained gait policy running at a fixed speed.

    python -m centaur_rl.tasks.Multi_Policy.tools.record_handoff \
        --task Multi-Gait_straight_flat_SGS-Walk-Refine \
        --checkpoint logs/.../model_9999.pt --speed 1.0 --out walk-up.pt

The policy runs `settle_s` seconds at `speed` (straight line, no SGS, no handoff
starts), then every `every` control steps the state of each env that has not fallen
is stored. Only envs that are still up at the end of the recording are kept. Saved states are yaw-free and
body-frame, so they can be replayed at any spawn position (mdp/reset.py).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch


def record(task: str, checkpoint: str, speed: float, out: str, num_envs: int = 512,
           settle_s: float = 3.0, record_s: float = 6.0, every: int = 5,
           max_states: int = 20000, device: str | None = None) -> dict:
  import mjlab.tasks  # noqa: F401
  import centaur_rl.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.lab_api.math import quat_apply_inverse, quat_inv, quat_mul, yaw_quat

  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = num_envs
  env_cfg.curriculum = {}
  env_cfg.observations["actor"].enable_corruption = False
  cmd_cfg = env_cfg.commands["gait"]
  cmd_cfg.speed_range = (speed, speed)
  cmd_cfg.resample_mid_episode = False
  cmd_cfg.resampling_time_range = (1e9, 1e9)
  cmd_cfg.turns = False  # banks are straight states (the refine env would otherwise turn)
  cmd_cfg.play_radius = None
  env_cfg.events["reset_gait_state"].params["bank_prob"] = 0.0
  env_cfg.episode_length_s = settle_s + record_s + 5.0

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)
  agent_cfg = load_rl_cfg(task)
  runner = load_runner_cls(task)(wrapped, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env.scene["robot"]
  alive = torch.ones(num_envs, dtype=torch.bool, device=device)
  settle_steps = int(settle_s / env.step_dt)
  total_steps = settle_steps + int(record_s / env.step_dt)
  keys = ("base_height", "quat_rp", "lin_vel_b", "ang_vel_b", "joint_pos", "joint_vel", "speed", "env_id")
  rows: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
  n_rec = 0
  obs = wrapped.get_observations()
  with torch.inference_mode():
    for step in range(total_steps):
      obs, _, dones, _ = wrapped.step(policy(obs))
      alive &= ~dones.bool()
      if step < settle_steps or step % every or not alive.any():
        continue
      ids = alive.nonzero().flatten()
      d = robot.data
      q = d.root_link_quat_w[ids]
      q_rp = quat_mul(quat_inv(yaw_quat(q)), q)  # remove heading
      rows["base_height"].append(d.root_link_pos_w[ids, 2] - env.scene.env_origins[ids, 2])
      rows["quat_rp"].append(q_rp)
      rows["lin_vel_b"].append(quat_apply_inverse(q, d.root_link_lin_vel_w[ids]))
      rows["ang_vel_b"].append(quat_apply_inverse(q, d.root_link_ang_vel_w[ids]))
      rows["joint_pos"].append(d.joint_pos[ids])
      rows["joint_vel"].append(d.joint_vel[ids])
      rows["speed"].append(torch.full((len(ids),), speed, device=device))
      rows["env_id"].append(ids)
      n_rec += len(ids)
      if n_rec >= 2 * max_states:  # headroom for the survivor filter below
        break
  env.close()

  if n_rec == 0:
    raise RuntimeError(f"no states recorded: every env fell within {settle_s}s at {speed} m/s")
  # Keep only envs that never fell: states shortly before a fall are not handoff states.
  keep = alive[torch.cat(rows.pop("env_id"))]
  if not keep.any():
    raise RuntimeError(f"every env fell during recording at {speed} m/s: policy not ready")
  bank = {k: torch.cat(v)[keep][:max_states].cpu() for k, v in rows.items()}
  gait = env_cfg.commands["gait"].gait_name
  bank["meta"] = {
    "gait": gait, "speed": speed, "task": task, "checkpoint": str(checkpoint),
    "survived_frac": float(alive.float().mean()), "num_states": len(bank["base_height"]),
  }
  Path(out).parent.mkdir(parents=True, exist_ok=True)
  torch.save(bank, out)
  print(f"[record_handoff] {bank['meta']['num_states']} states of {gait} @ {speed} m/s -> {out} "
        f"(survived {bank['meta']['survived_frac']:.0%})")
  return bank["meta"]


def main(argv: list[str] | None = None) -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--task", required=True)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--speed", type=float, required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--num-envs", type=int, default=512)
  p.add_argument("--settle-s", type=float, default=3.0)
  p.add_argument("--record-s", type=float, default=6.0)
  p.add_argument("--every", type=int, default=5)
  p.add_argument("--max-states", type=int, default=20000)
  p.add_argument("--device", default=None)
  a = p.parse_args(argv)
  record(a.task, a.checkpoint, a.speed, a.out, a.num_envs, a.settle_s, a.record_s,
         a.every, a.max_states, a.device)


if __name__ == "__main__":
  sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
  main()
