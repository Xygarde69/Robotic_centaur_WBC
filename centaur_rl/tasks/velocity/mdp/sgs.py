"""SGS: Success-Guided Sampling (Zhang, Guaman Castro, Yin, Scalise et al., arXiv:2610.12465).

Paper summary (Sec. 3.2, App. A.1, B.2):
  * Before training, fix a set of N task configurations tau_i = (s0, g, e).
  * Keep the last H boolean outcomes per configuration; p_i = their mean.
  * Score with a Beta kernel in mode/concentration form (target t, concentration kappa):
        w_i = (p_i + eps)^(kappa t) * (1 - p_i + eps)^(kappa (1 - t))
        w_i <- max(w_i, eps);  l_i = log(w_i + eps)
  * At every episode reset sample the next configuration i.i.d. from softmax(l_i / T_eff),
    T_eff = max(T, 1).
  * Locomotion hyper-parameters: N ~ 104k, H = 100, T = 2, t = 0.66, kappa = 5, eps = 1e-8.

Adaptation to this velocity-tracking task (the paper does goal reaching):
  * Configuration = (terrain row/level, terrain col/type, spawn pose, velocity command).
    Sizes are set in config/centaur/env_cfgs.py (scaled down from the paper's 104k to
    suit ~1K parallel envs; see the comment there).
  * The command is held fixed for the whole episode (it is the episode's "goal").
  * "Command-success" = episode ran to time-out (no failure termination) AND its mean
    linear (xy) and yaw-rate tracking errors are below thresholds.
  * SGS replaces the hand-designed terrain-level and command-velocity curricula.

Wiring (all in this module):
  * `SuccessGuidedSampling`: curriculum term. mjlab calls curriculum terms first in
    `_reset_idx`, before events and the command reset, so it records the finished
    episode's outcome, samples the next config and moves the env to its terrain tile.
  * `reset_root_state_sgs`: reset event placing the robot at the config's spawn pose.
  * `SgsVelocityCommand`: command term that uses the config's command and accumulates
    the per-episode tracking errors used for the success test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class SuccessGuidedSampling:
  """Curriculum term implementing SGS over (terrain tile, spawn, command) configurations."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self._env = env
    self.device = env.device
    self.command_name: str = p["command_name"]
    self.target: float = p.get("target", 0.66)
    self.kappa: float = p.get("kappa", 5.0)
    self.temperature: float = max(p.get("temperature", 2.0), 1.0)  # T_eff = max(T, 1)
    self.eps: float = p.get("eps", 1e-8)
    self.history: int = p.get("history", 100)
    self.max_err_xy: float = p.get("max_err_xy", 0.3)
    self.max_err_yaw: float = p.get("max_err_yaw", 0.3)
    self.min_episode_frac: float = p.get("min_episode_frac", 0.5)

    gen = torch.Generator(device="cpu").manual_seed(p.get("seed", 0))

    # Terrain tiles (rows = difficulty level, cols = terrain type), if any.
    terrain = env.scene.terrain
    self.terrain = terrain
    self.has_tiles = terrain is not None and terrain.terrain_origins is not None
    self.rows, self.cols = (
      terrain.terrain_origins.shape[:2] if self.has_tiles else (1, 1)
    )

    # Spawn poses: (dx, dy, yaw) offsets from the tile/env origin.
    S = p.get("num_spawns", 8)
    sr = p.get("spawn_range", {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)})
    self.spawns = torch.stack(
      [_uniform(gen, S, *sr["x"]), _uniform(gen, S, *sr["y"]), _uniform(gen, S, *sr["yaw"])],
      dim=-1,
    ).to(self.device)

    # Fixed velocity commands (lin_x, lin_y, ang_z). A fraction is pure standing.
    K = p.get("num_commands", 64)
    cr = p["command_ranges"]
    cmds = torch.stack(
      [_uniform(gen, K, *cr["lin_vel_x"]), _uniform(gen, K, *cr["lin_vel_y"]),
       _uniform(gen, K, *cr["ang_vel_z"])],
      dim=-1,
    )
    n_stand = int(math.ceil(p.get("standing_frac", 0.05) * K))
    cmds[:n_stand] = 0.0
    cmds[torch.linalg.norm(cmds, dim=-1) < 0.1] = 0.0  # same dead-band as the task
    self.commands = cmds.to(self.device)

    self.num_spawns, self.num_commands = S, K
    self.num_configs = self.rows * self.cols * S * K

    # Rolling outcome history (circular buffer of H booleans per configuration).
    self.hist = torch.zeros(self.num_configs, self.history, dtype=torch.bool, device=self.device)
    self.hist_ptr = torch.zeros(self.num_configs, dtype=torch.long, device=self.device)
    self.hist_count = torch.zeros(self.num_configs, dtype=torch.long, device=self.device)

    # Per-env assignment.
    self.env_cfg = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
    self.assigned = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

    env.sgs = self  # shared with the reset event and the command term
    print(
      f"[SGS] {self.num_configs} configs = {self.rows}x{self.cols} tiles x {S} spawns x {K} "
      f"commands | t={self.target} kappa={self.kappa} T={self.temperature} H={self.history}"
    )

  # ------------------------------------------------------------------ config decoding
  def decode(self, idx: torch.Tensor):
    K, S, C = self.num_commands, self.num_spawns, self.cols
    k = idx % K
    s = (idx // K) % S
    c = (idx // (K * S)) % C
    r = idx // (K * S * C)
    return r, c, s, k

  def spawn_of(self, env_ids: torch.Tensor) -> torch.Tensor:
    return self.spawns[self.decode(self.env_cfg[env_ids])[2]]

  def command_of(self, env_ids: torch.Tensor) -> torch.Tensor:
    return self.commands[self.decode(self.env_cfg[env_ids])[3]]

  # ------------------------------------------------------------------ SGS core
  def success_rates(self) -> torch.Tensor:
    return self.hist.sum(-1).float() / self.hist_count.clamp(min=1).float()

  def sampling_probs(self) -> torch.Tensor:
    p, t, kap, eps = self.success_rates(), self.target, self.kappa, self.eps
    w = (p + eps) ** (kap * t) * (1.0 - p + eps) ** (kap * (1.0 - t))  # Eq. 1
    w = torch.clamp(w, min=eps)  # App. B.2 floor
    logits = torch.log(w + eps)
    return torch.softmax(logits / self.temperature, dim=0)  # Eq. 2

  def _record(self, env_ids: torch.Tensor) -> tuple[int, int]:
    env = self._env
    tm = env.termination_manager
    cmd: SgsVelocityCommand = env.command_manager.get_term(self.command_name)
    max_len = env.max_episode_length

    ep_len = env.episode_length_buf[env_ids]
    failed = tm.terminated[env_ids]
    timed_out = tm.time_outs[env_ids] & ~failed
    valid = self.assigned[env_ids] & (
      failed | (timed_out & (ep_len >= self.min_episode_frac * max_len))
    )
    if not valid.any():
      return 0, 0

    err_xy, err_yaw = cmd.episode_errors(env_ids)
    success = timed_out & (err_xy < self.max_err_xy) & (err_yaw < self.max_err_yaw)

    idx = self.env_cfg[env_ids][valid]
    ok = success[valid]
    pos = self.hist_ptr[idx]
    self.hist[idx, pos] = ok
    self.hist_ptr[idx] = (pos + 1) % self.history
    self.hist_count[idx] = torch.clamp(self.hist_count[idx] + 1, max=self.history)
    return int(valid.sum()), int(ok.sum())

  def __call__(self, env: ManagerBasedRlEnv, env_ids, **_params) -> dict[str, torch.Tensor]:
    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=self.device)[env_ids]
    n_rec, n_ok = self._record(env_ids)

    # Sample next configurations.
    probs = self.sampling_probs()
    new = torch.multinomial(probs, len(env_ids), replacement=True)
    self.env_cfg[env_ids] = new
    self.assigned[env_ids] = True

    r, c, _, _ = self.decode(new)
    if self.has_tiles:
      t = self.terrain
      t.terrain_levels[env_ids] = r
      t.terrain_types[env_ids] = c
      t.env_origins[env_ids] = t.terrain_origins[r, c]

    seen = self.hist_count > 0
    sr = self.success_rates()
    frontier = seen & (sr > 0.0) & (sr < 1.0)
    return {
      "episode_success": torch.tensor(n_ok / max(n_rec, 1)),
      "configs_seen_frac": seen.float().mean(),
      "configs_frontier_frac": frontier.float().mean(),
      "mean_success_seen": sr[seen].mean() if seen.any() else torch.tensor(0.0),
      "sampled_terrain_level": r.float().mean(),
      "sampled_cmd_speed": torch.linalg.norm(self.commands[self.decode(new)[3], :2], dim=-1).mean(),
      "probs_max_over_uniform": probs.max() * self.num_configs,
    }

  def reset(self, env_ids=None) -> None:  # class-term hook; nothing per-episode to clear
    pass


def _uniform(gen: torch.Generator, n: int, lo: float, hi: float) -> torch.Tensor:
  return torch.rand(n, generator=gen) * (hi - lo) + lo


# ---------------------------------------------------------------------- reset event
def reset_root_state_sgs(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Place the robot at its SGS configuration's spawn pose (origin + dx, dy, yaw), at rest."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  sgs: SuccessGuidedSampling = env.sgs
  spawn = sgs.spawn_of(env_ids)

  root = asset.data.default_root_state[env_ids].clone()
  root[:, 0:3] += env.scene.env_origins[env_ids]
  root[:, 0:2] += spawn[:, 0:2]
  zeros = torch.zeros_like(spawn[:, 2])
  root[:, 3:7] = quat_mul(root[:, 3:7], quat_from_euler_xyz(zeros, zeros, spawn[:, 2]))
  root[:, 7:13] = 0.0
  asset.write_root_state_to_sim(root, env_ids)


# ---------------------------------------------------------------------- command term
class SgsVelocityCommand(UniformVelocityCommand):
  """Velocity command fixed per episode to the SGS configuration's command."""

  cfg: SgsVelocityCommandCfg

  def __init__(self, cfg: SgsVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.ep_err_xy = torch.zeros(self.num_envs, device=self.device)
    self.ep_err_yaw = torch.zeros(self.num_envs, device=self.device)
    self.ep_steps = torch.zeros(self.num_envs, device=self.device)

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self.ep_err_xy += torch.norm(
      self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
    )
    self.ep_err_yaw += torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
    self.ep_steps += 1

  def episode_errors(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = self.ep_steps[env_ids].clamp(min=1)
    return self.ep_err_xy[env_ids] / n, self.ep_err_yaw[env_ids] / n

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    sgs: SuccessGuidedSampling | None = getattr(self._env, "sgs", None)
    if sgs is None:  # sampler not built yet (manager construction): fall back to uniform
      super()._resample_command(env_ids)
    else:
      self.vel_command_b[env_ids] = sgs.command_of(env_ids)
      self.is_standing_env[env_ids] = False
      self.is_heading_env[env_ids] = False
    self.ep_err_xy[env_ids] = 0.0
    self.ep_err_yaw[env_ids] = 0.0
    self.ep_steps[env_ids] = 0.0


@dataclass(kw_only=True)
class SgsVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> SgsVelocityCommand:
    return SgsVelocityCommand(self, env)
