"""Episode start states: spawn moving at the commanded speed, or from a handoff bank.

`GaitStateReset` (reset event) does, per resetting env:
  1. draws the episode's first commanded speed (GaitCommand.draw_episode_speed), so the
     spawn can match it before the command manager resets;
  2. with probability `bank_prob`, starts the episode from a state recorded from a
     neighbouring gait's policy (handoff bank: base height, roll/pitch, body-frame
     velocities, joint positions and velocities); otherwise
  3. spawns in the default pose (+ joint noise) at rest (walk) or already moving forward
     at the commanded speed (every other gait: they never accelerate from zero).

Banks are recorded with tools/record_handoff.py and loaded either from the event's
`bank_paths` param or at runtime by the runner (`env.gait_reset.load_banks`), which is
how the pipeline wires them per stage.

Bank file (torch.save dict):
  base_height [N], quat_rp [N, 4] (yaw removed, wxyz), lin_vel_b [N, 3], ang_vel_b [N, 3],
  joint_pos [N, J], joint_vel [N, J], speed [N] (source policy's command), meta {...}.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

  from .command import GaitCommand

BANK_KEYS = ("base_height", "quat_rp", "lin_vel_b", "ang_vel_b", "joint_pos", "joint_vel")


class GaitStateReset:
  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self._env = env
    self.asset_cfg: SceneEntityCfg = p.get("asset_cfg", SceneEntityCfg("robot"))
    self.command_name: str = p["command_name"]
    self.bank: dict[str, torch.Tensor] | None = None
    self.bank_sources: list[str] = []
    self.bank_prob: float = 0.0
    self.default_bank_prob: float = float(p.get("bank_prob", 0.0))  # this phase's share
    if p.get("bank_paths"):
      self.load_banks(p["bank_paths"], p.get("bank_prob", 0.0))
    env.gait_reset = self

  # ------------------------------------------------------------------ banks
  def load_banks(self, paths: list[str] | tuple[str, ...], prob: float | None = None) -> None:
    """Load and concatenate handoff banks; `prob` = share of resets drawn from them
    (None = the phase default from the env config)."""
    prob = self.default_bank_prob if prob is None else prob
    parts = []
    for path in paths:
      data = torch.load(Path(path), map_location=self._env.device, weights_only=False)
      missing = [k for k in BANK_KEYS if k not in data]
      if missing:
        raise KeyError(f"handoff bank {path} lacks {missing}")
      parts.append(data)
      meta = data.get("meta", {})
      print(
        f"[GaitStateReset] bank {path}: {len(data['base_height'])} states from "
        f"{meta.get('gait', '?')} at {meta.get('speed', '?')} m/s"
      )
    if not parts:
      self.bank, self.bank_prob = None, 0.0
      return
    self.bank = {k: torch.cat([d[k].to(self._env.device) for d in parts]) for k in BANK_KEYS}
    asset: Entity = self._env.scene[self.asset_cfg.name]
    if self.bank["joint_pos"].shape[1] != asset.data.default_joint_pos.shape[1]:
      raise ValueError("handoff bank joint count does not match the robot")
    self.bank_sources = [str(p) for p in paths]
    self.bank_prob = float(prob)

  # ------------------------------------------------------------------ event
  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    spawn_from_rest: bool = False,
    joint_noise: float = 0.0,
    bank_paths: tuple[str, ...] = (),
    bank_prob: float = 0.0,
    asset_cfg: SceneEntityCfg | None = None,
  ) -> None:
    del bank_paths, bank_prob, asset_cfg  # Resolved in __init__ / load_banks.
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    asset: Entity = env.scene[self.asset_cfg.name]
    cmd: GaitCommand = env.command_manager.get_term(command_name)
    speed = cmd.draw_episode_speed(env_ids)
    n = len(env_ids)
    cmd.from_rest[env_ids] = spawn_from_rest  # bank starts clear it below

    root = asset.data.default_root_state[env_ids].clone()
    root[:, 0:3] += env.scene.env_origins[env_ids]
    root[:, 7:13] = 0.0
    if not spawn_from_rest:
      lin_b = torch.zeros(n, 3, device=env.device)
      lin_b[:, 0] = speed
      root[:, 7:10] = quat_apply(root[:, 3:7], lin_b)
      # On a turn the robot starts on its arc: yaw rate sign * v / r (0 when straight).
      ang_b = torch.zeros(n, 3, device=env.device)
      ang_b[:, 2] = cmd._next_sign[env_ids] * speed / cmd._next_radius[env_ids]
      root[:, 10:13] = quat_apply(root[:, 3:7], ang_b)

    jpos = asset.data.default_joint_pos[env_ids].clone()
    jvel = torch.zeros_like(jpos)
    if joint_noise > 0:
      jpos += (2.0 * torch.rand_like(jpos) - 1.0) * joint_noise
    limits = asset.data.soft_joint_pos_limits[env_ids]
    jpos = jpos.clamp(limits[..., 0], limits[..., 1])

    if self.bank is not None and self.bank_prob > 0:
      use = torch.rand(n, device=env.device) < self.bank_prob
      k = int(use.sum())
      if k:
        idx = torch.randint(0, len(self.bank["base_height"]), (k,), device=env.device)
        b = {key: v[idx] for key, v in self.bank.items()}
        sub = root[use]
        sub[:, 2] = env.scene.env_origins[env_ids[use], 2] + b["base_height"]
        sub[:, 3:7] = b["quat_rp"]
        sub[:, 7:10] = quat_apply(b["quat_rp"], b["lin_vel_b"])
        sub[:, 10:13] = quat_apply(b["quat_rp"], b["ang_vel_b"])
        root[use] = sub
        jpos[use] = b["joint_pos"]
        jvel[use] = b["joint_vel"]
        cmd.from_rest[env_ids[use]] = False
      env.extras.setdefault("log", {})["Episode/handoff_start_frac"] = torch.tensor(k / n)

    asset.write_root_state_to_sim(root, env_ids)
    asset.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)
