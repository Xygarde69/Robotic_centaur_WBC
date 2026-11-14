"""SGS: Success-Guided Sampling (Zhang, Guaman Castro, Yin, Scalise et al., arXiv:2610.12465).

Paper summary (Sec. 3.2, App. A.1, B.2):
  * Before training, fix a set of N task configurations tau_i = (s0, g, e). For locomotion
    it is a product of axes: terrain type x terrain level x start state x goal state.
  * Keep the last H boolean outcomes per configuration; p_i = their mean.
  * Score with a Beta kernel in mode/concentration form (target t, concentration kappa):
        w_i = (p_i + eps)^(kappa t) * (1 - p_i + eps)^(kappa (1 - t))
        w_i <- max(w_i, eps);  l_i = log(w_i + eps)
  * At every episode reset sample the next configuration i.i.d. from softmax(l_i / T_eff),
    T_eff = max(T, 1).
  * Locomotion hyper-parameters: N ~ 104k, H = 100, T = 2, t = 0.66, kappa = 5, eps = 1e-8.

Adaptation to this velocity-tracking task (the paper does goal reaching):
  * Configuration = terrain level (row) x terrain type (col) x one cell of the command
    parameter grid. The command parameters are named axes, each split into bins
    (`axes` param: name -> (low, high, bins)):
        lin_vel_x, lin_vel_y, ang_vel_z   the commanded velocity itself
    Every episode draws its parameters uniformly inside its configuration's cell, so the
    whole continuous range is trained while the cell keeps the success statistics.
  * The spawn (position jitter, heading) is drawn fresh at every reset and is not a
    configuration axis: the robot starts on the tile's centre and the terrains are
    symmetric about it, so the heading does not change the difficulty.
  * The command is held fixed for the whole episode (it is the episode's "goal").
  * Success is decided on the configuration's own tile. The robot covers 20-40 m in an
    episode but a tile is 8 m wide, so on a straight command it is on other tiles (other
    types and levels) for most of the episode. The outcome is therefore frozen when the
    robot leaves its tile:
        left the tile          -> success if it tracked the command while on it
        fell while on the tile -> failure
        timed out on the tile  -> success if it tracked the command (turning commands)
    What happens after leaving the tile still trains the policy but is not attributed
    to the configuration. Without tiles (flat) the whole episode counts.
  * "Tracked the command": the mean velocity over the counted steps is within
    max(max_err_xy, max_err_xy_rel * |cmd_xy|) m/s and max_err_yaw rad/s of the command.
    The mean of the signed error is used, so the velocity ripple inside each stride
    (inherent to the gait) is not counted as a tracking error, and the first settle_time
    seconds (accelerating from rest) are left out.
  * Optionally a fixed share of episodes (standing_prob) is a standing episode outside
    the configuration set, so standing keeps being practised whatever SGS samples.
  * SGS replaces the hand-designed terrain-level and command-velocity curricula.

Wiring (all in this module):
  * `SuccessGuidedSampling`: curriculum term. mjlab calls curriculum terms first in
    `_reset_idx`, before events and the command reset, so it records the finished
    episode's outcome, samples the next config and moves the env to its terrain tile.
  * `reset_root_state_sgs`: reset event placing the robot at the episode's spawn pose.
  * `SgsVelocityCommand`: command term that turns the episode's parameters into the
    command and accumulates the tracking errors used for the success test.
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
  """Curriculum term implementing SGS over (terrain tile, command cell) configurations."""

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
    self.max_err_xy_rel: float = p.get("max_err_xy_rel", 0.0)
    self.max_err_yaw: float = p.get("max_err_yaw", 0.3)
    self.min_episode_frac: float = p.get("min_episode_frac", 0.5)
    self.standing_prob: float = p.get("standing_prob", 0.0)
    self.spawn_range: dict[str, tuple[float, float]] = p.get(
      "spawn_range", {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)}
    )

    # Terrain tiles (rows = difficulty level, cols = terrain type), if any.
    terrain = env.scene.terrain
    self.terrain = terrain
    self.has_tiles = terrain is not None and terrain.terrain_origins is not None
    self.rows, self.cols = (
      terrain.terrain_origins.shape[:2] if self.has_tiles else (1, 1)
    )
    # Half-extent (x, y) of the area around the tile origin that counts as "on the tile".
    if self.has_tiles:
      size = terrain.cfg.terrain_generator.size
      margin = p.get("tile_margin", 0.25)
      self.tile_half = torch.tensor(
        [0.5 * size[0] - margin, 0.5 * size[1] - margin], device=self.device
      )
    else:
      self.tile_half = torch.full((2,), float("inf"), device=self.device)

    # Command parameter grid: one axis per parameter, (low, high, bins).
    axes: dict[str, tuple[float, float, int]] = p["axes"]
    self.axis_names = list(axes)
    self.axis_lo = torch.tensor([axes[n][0] for n in self.axis_names], device=self.device)
    self.axis_hi = torch.tensor([axes[n][1] for n in self.axis_names], device=self.device)
    self.axis_bins = torch.tensor(
      [int(axes[n][2]) for n in self.axis_names], dtype=torch.long, device=self.device
    )
    # Axes whose middle bin is pinned to the middle of the range (e.g. exactly straight).
    snap = set(p.get("snap_center", ()))
    for n in snap:
      assert axes[n][2] % 2 == 1, f"snap_center axis '{n}' needs an odd number of bins"
    self.axis_snap = torch.tensor([n in snap for n in self.axis_names], device=self.device)
    self.num_cells = int(torch.prod(self.axis_bins))
    self.num_configs = self.rows * self.cols * self.num_cells

    # Rolling outcome history (circular buffer of H booleans per configuration).
    self.hist = torch.zeros(self.num_configs, self.history, dtype=torch.bool, device=self.device)
    self.hist_ptr = torch.zeros(self.num_configs, dtype=torch.long, device=self.device)
    self.hist_count = torch.zeros(self.num_configs, dtype=torch.long, device=self.device)

    # Per-env assignment for the running episode.
    n = env.num_envs
    self.env_cfg = torch.zeros(n, dtype=torch.long, device=self.device)
    self.assigned = torch.zeros(n, dtype=torch.bool, device=self.device)  # outcome is recorded
    self.env_standing = torch.zeros(n, dtype=torch.bool, device=self.device)
    self.env_values = torch.zeros(n, len(self.axis_names), device=self.device)
    self.env_spawn = torch.zeros(n, 3, device=self.device)  # dx, dy, yaw

    env.sgs = self  # shared with the reset event and the command term
    grid = " x ".join(f"{n}:{int(b)}" for n, b in zip(self.axis_names, self.axis_bins))
    print(
      f"[SGS] {self.num_configs} configs = {self.rows}x{self.cols} tiles x ({grid}) | "
      f"t={self.target} kappa={self.kappa} T={self.temperature} H={self.history}"
    )

  # ------------------------------------------------------------------ config decoding
  def decode(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Config index -> (terrain row, terrain col, bin index per axis [n, A])."""
    cell = idx % self.num_cells
    tile = idx // self.num_cells
    bins = []
    for b in self.axis_bins.tolist():
      bins.append(cell % b)
      cell = cell // b
    return tile // self.cols, tile % self.cols, torch.stack(bins, dim=-1)

  def _draw_values(self, bins: torch.Tensor) -> torch.Tensor:
    """Uniform draw inside each bin, [n, A]."""
    width = (self.axis_hi - self.axis_lo) / self.axis_bins
    u = torch.rand(bins.shape, device=self.device)
    pinned = self.axis_snap & (bins == (self.axis_bins - 1) // 2)
    u = torch.where(pinned, 0.5, u)
    return self.axis_lo + (bins + u) * width

  def values_of(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """The running episode's command parameters, by axis name."""
    v = self.env_values[env_ids]
    return {n: v[:, i] for i, n in enumerate(self.axis_names)}

  # ------------------------------------------------------------------ SGS core
  def success_rates(self) -> torch.Tensor:
    return self.hist.sum(-1).float() / self.hist_count.clamp(min=1).float()

  def sampling_probs(self) -> torch.Tensor:
    p, t, kap, eps = self.success_rates(), self.target, self.kappa, self.eps
    w = (p + eps) ** (kap * t) * (1.0 - p + eps) ** (kap * (1.0 - t))  # Eq. 1
    w = torch.clamp(w, min=eps)  # App. B.2 floor
    logits = torch.log(w + eps)
    return torch.softmax(logits / self.temperature, dim=0)  # Eq. 2

  def _record(self, env_ids: torch.Tensor) -> tuple[int, int, int]:
    env = self._env
    tm = env.termination_manager
    cmd: SgsVelocityCommand = env.command_manager.get_term(self.command_name)

    ep_len = env.episode_length_buf[env_ids]
    failed = tm.terminated[env_ids]
    timed_out = tm.time_outs[env_ids] & ~failed
    err_xy, err_yaw, counted, left = cmd.episode_outcome(env_ids)
    # Time-outs of truncated (randomized-start) episodes that stayed on the tile are skipped.
    full = timed_out & (ep_len >= self.min_episode_frac * env.max_episode_length)
    valid = self.assigned[env_ids] & (left | failed | full)
    if not valid.any():
      return 0, 0, 0

    speed = torch.norm(cmd.vel_command_b[env_ids, :2], dim=-1)
    tol_xy = torch.clamp(self.max_err_xy_rel * speed, min=self.max_err_xy)
    tracked = counted & (err_xy < tol_xy) & (err_yaw < self.max_err_yaw)
    success = (left | timed_out) & tracked

    idx = self.env_cfg[env_ids][valid]
    ok = success[valid]
    pos = self.hist_ptr[idx]
    self.hist[idx, pos] = ok
    self.hist_ptr[idx] = (pos + 1) % self.history
    self.hist_count[idx] = torch.clamp(self.hist_count[idx] + 1, max=self.history)
    return int(valid.sum()), int(ok.sum()), int((left & valid).sum())

  def __call__(self, env: ManagerBasedRlEnv, env_ids, **_params) -> dict[str, torch.Tensor]:
    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=self.device)[env_ids]
    n_rec, n_ok, n_left = self._record(env_ids)

    # Sample next configurations and this episode's parameters inside their cells.
    probs = self.sampling_probs()
    new = torch.multinomial(probs, len(env_ids), replacement=True)
    r, c, bins = self.decode(new)
    values = self._draw_values(bins)
    standing = torch.rand(len(env_ids), device=self.device) < self.standing_prob
    self.env_cfg[env_ids] = new
    self.env_values[env_ids] = values
    self.env_standing[env_ids] = standing
    self.assigned[env_ids] = ~standing
    sr = self.spawn_range
    u = torch.rand(len(env_ids), 3, device=self.device)
    lo = torch.tensor([sr["x"][0], sr["y"][0], sr["yaw"][0]], device=self.device)
    hi = torch.tensor([sr["x"][1], sr["y"][1], sr["yaw"][1]], device=self.device)
    self.env_spawn[env_ids] = lo + u * (hi - lo)

    if self.has_tiles:
      t = self.terrain
      t.terrain_levels[env_ids] = r
      t.terrain_types[env_ids] = c
      t.env_origins[env_ids] = t.terrain_origins[r, c]

    seen = self.hist_count > 0
    rates = self.success_rates()
    frontier = seen & (rates > 0.0) & (rates < 1.0)
    out = {
      "episode_success": torch.tensor(n_ok / max(n_rec, 1)),
      "episode_left_tile_frac": torch.tensor(n_left / max(n_rec, 1)),
      "configs_seen_frac": seen.float().mean(),
      "configs_frontier_frac": frontier.float().mean(),
      "mean_success_seen": rates[seen].mean() if seen.any() else torch.tensor(0.0),
      "sampled_terrain_level": r.float().mean(),
      "probs_max_over_uniform": probs.max() * self.num_configs,
    }
    # What SGS is currently asking for along each command axis (magnitude for signed axes).
    for i, name in enumerate(self.axis_names):
      v = values[:, i].abs() if self.axis_lo[i] < 0 else values[:, i]
      out[f"sampled_{name}"] = v.mean()
    if self.has_tiles:
      # Estimated success by terrain level (row), easiest and hardest thirds.
      per_row = rates.view(self.rows, -1)
      seen_row = seen.view(self.rows, -1)
      k = max(self.rows // 3, 1)
      for tag, sl in (("easy", slice(0, k)), ("hard", slice(self.rows - k, self.rows))):
        m = seen_row[sl]
        out[f"success_{tag}_levels"] = per_row[sl][m].mean() if m.any() else torch.tensor(0.0)
    return out

  def reset(self, env_ids=None) -> None:  # class-term hook; nothing per-episode to clear
    pass


# ---------------------------------------------------------------------- reset event
def reset_root_state_sgs(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Place the robot at the episode's spawn pose (tile origin + dx, dy, yaw), at rest."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  sgs: SuccessGuidedSampling = env.sgs
  spawn = sgs.env_spawn[env_ids]

  root = asset.data.default_root_state[env_ids].clone()
  root[:, 0:3] += env.scene.env_origins[env_ids]
  root[:, 0:2] += spawn[:, 0:2]
  zeros = torch.zeros_like(spawn[:, 2])
  root[:, 3:7] = quat_mul(root[:, 3:7], quat_from_euler_xyz(zeros, zeros, spawn[:, 2]))
  root[:, 7:13] = 0.0
  asset.write_root_state_to_sim(root, env_ids)


# ---------------------------------------------------------------------- command term
class SgsVelocityCommand(UniformVelocityCommand):
  """Velocity command fixed per episode, drawn from the SGS configuration's cell."""

  cfg: SgsVelocityCommandCfg

  def __init__(self, cfg: SgsVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    # Signed error sums (x, y, yaw) over the counted steps of the running episode.
    self.ep_err = torch.zeros(self.num_envs, 3, device=self.device)
    self.ep_steps = torch.zeros(self.num_envs, device=self.device)
    self.left_tile = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def _update_metrics(self) -> None:
    super()._update_metrics()
    env = self._env
    sgs: SuccessGuidedSampling | None = getattr(env, "sgs", None)
    if sgs is not None and sgs.has_tiles:
      offset = self.robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
      self.left_tile |= (offset.abs() > sgs.tile_half).any(dim=-1)
    settled = env.episode_length_buf * env.step_dt >= self.cfg.settle_time
    count = (settled & ~self.left_tile).float()
    err = torch.cat(
      [
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2],
        (self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]).unsqueeze(-1),
      ],
      dim=-1,
    )
    self.ep_err += err * count.unsqueeze(-1)
    self.ep_steps += count

  def episode_outcome(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(|mean xy error|, |mean yaw error|, any step counted, left its tile) per env."""
    n = self.ep_steps[env_ids]
    mean = self.ep_err[env_ids] / n.clamp(min=1).unsqueeze(-1)
    return torch.norm(mean[:, :2], dim=-1), mean[:, 2].abs(), n > 0, self.left_tile[env_ids]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    sgs: SuccessGuidedSampling | None = getattr(self._env, "sgs", None)
    if sgs is None:  # sampler not built yet (manager construction): fall back to uniform
      super()._resample_command(env_ids)
    else:
      v = sgs.values_of(env_ids)
      cmd = torch.stack([v["lin_vel_x"], v["lin_vel_y"], v["ang_vel_z"]], dim=-1)
      self.vel_command_b[env_ids] = cmd * (~sgs.env_standing[env_ids]).unsqueeze(-1)
      self.is_standing_env[env_ids] = sgs.env_standing[env_ids]
      self.is_heading_env[env_ids] = False
    self.ep_err[env_ids] = 0.0
    self.ep_steps[env_ids] = 0.0
    self.left_tile[env_ids] = False


@dataclass(kw_only=True)
class SgsVelocityCommandCfg(UniformVelocityCommandCfg):
  settle_time: float = 1.0  # s after reset left out of the success test (acceleration)

  def build(self, env: ManagerBasedRlEnv) -> SgsVelocityCommand:
    return SgsVelocityCommand(self, env)
