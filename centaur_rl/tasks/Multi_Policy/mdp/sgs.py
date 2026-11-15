"""Speed (and, in refine, turn radius) SGS for one gait on flat ground.

Success-Guided Sampling (arXiv:2610.12465). Ladder: one axis, the forward speed, split
into `num_bins` bins over the gait's SGS range; straight line. Refine: a grid of speed
bins x turn-radius bins (`turn_radii`, widest first, from 2 km = straight in practice).
Each episode draws its speed uniformly inside its speed bin; its radius is the bin's.
No terrain tiles.

Two modes:

  ladder  V_current = the frontier bin. A share `frontier_prob` of the episodes is
          sampled AT the frontier; the rest is SGS-weighted (Beta kernel) over the
          bins already unlocked (replay of slower speeds). The frontier advances when
          its success rate over its last `history` outcomes reaches
          `advance_threshold` (with at least `min_outcomes` of them), and retreats if
          it falls below `retreat_threshold` after advancing (optional).
          => speeds climb from v_min to v_max as each speed is mastered, instead of on
          a fixed iteration schedule.

  refine  every speed bin is unlocked. Turn radius is a difficulty axis: each speed bin
          s has a radius frontier f_s (starts at the widest radius) that moves one bin
          tighter when config (s, f_s) reaches `advance_threshold` success over
          `min_outcomes`, down to the tightest radius with v_hi(s)^2 / r <=
          `turn_max_lat_accel` (tighter configs are infeasible and never sampled).
          `refine_frontier_prob` of the episodes are drawn at a frontier config (uniform
          speed bin), the rest SGS-weighted over all unlocked configs (r >= r_{f_s}).
          The command term resamples the speed mid-episode (up and down) among the speed
          bins that have the episode's radius unlocked; the radius stays.

Success of an episode (decided at its reset):
  ran to time-out (no failure termination), and over the counted steps (after
  settle_time) |mean(v_x - v_cmd)| <= max(err_abs, err_rel * mean|v_cmd|) and
  |mean(yaw rate)| <= err_yaw. With timing_tol set (walk) the footfalls must also keep the
  clock's spacing: at least one stride, and the mean over strides of the worst deviation
  of the left/right and hind->fore touchdown spacings <= timing_tol (cycle fractions). Mean of the SIGNED error: the in-stride velocity ripple
  of the flight gaits is not a tracking error. On turns the yaw test is on the signed
  yaw-rate error: |mean(w_z - w_cmd)| <= max(err_yaw, err_yaw_rel * mean|w_cmd|).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

  from .command import GaitCommand


class SpeedSGS:
  """Curriculum term: records outcomes and samples the next episode's speed bin."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self._env = env
    self.device = env.device
    self.command_name: str = p["command_name"]
    self.mode: str = p.get("mode", "ladder")
    assert self.mode in ("ladder", "refine"), self.mode

    lo, hi = p["speed_range"]
    self.num_bins = int(p["num_bins"])
    edges = torch.linspace(lo, hi, self.num_bins + 1, device=self.device)
    self.bin_lo, self.bin_hi = edges[:-1], edges[1:]

    # Paper's locomotion hyper-parameters.
    self.target = p.get("target", 0.66)
    self.kappa = p.get("kappa", 5.0)
    self.temperature = max(p.get("temperature", 2.0), 1.0)
    self.eps = p.get("eps", 1e-8)
    self.history = int(p.get("history", 100))

    # Success test.
    self.err_abs = p.get("err_abs", 0.1)
    self.err_rel = p.get("err_rel", 0.1)
    self.err_yaw = p.get("err_yaw", 0.2)
    self.min_counted_time = p.get("min_counted_time", 2.0)  # s, after settle_time
    self.timing_tol = p.get("timing_tol")  # footfall spacing tolerance (None = off)
    self.err_yaw_rel = p.get("err_yaw_rel", 0.2)  # turns: share of the commanded yaw rate

    # Ladder.
    self.frontier_prob = p.get("frontier_prob", 0.8)
    self.advance_threshold = p.get("advance_threshold", 0.8)
    self.retreat_threshold = p.get("retreat_threshold", 0.0)  # 0 disables retreating
    self.min_outcomes = int(p.get("min_outcomes", 50))
    self.frontier = 0 if self.mode == "ladder" else self.num_bins - 1
    self.frontier_max = self.frontier

    # Turn-radius axis (refine only). Config id = speed_bin * num_radii + radius_bin.
    radii = tuple(p.get("turn_radii") or ()) if self.mode == "refine" else ()
    self.turns = len(radii) > 0
    self.radii = torch.tensor(radii or (math.inf,), device=self.device)
    self.num_radii = len(self.radii)
    a_lat = p.get("turn_max_lat_accel", math.inf)
    # feasible[s, k]: the fastest speed of bin s can hold radius k within the lateral cap.
    self.feasible = self.bin_hi.square().unsqueeze(1) / self.radii.unsqueeze(0) <= a_lat
    self.feasible[:, 0] = True
    self.radius_max_bin = self.feasible.long().cumprod(dim=1).sum(dim=1) - 1  # [S]
    self.radius_frontier = torch.zeros(self.num_bins, dtype=torch.long, device=self.device)
    self.refine_frontier_prob = p.get("refine_frontier_prob", 0.5)
    self.num_configs = self.num_bins * self.num_radii

    self.hist = torch.zeros(self.num_configs, self.history, dtype=torch.bool, device=self.device)
    self.hist_ptr = torch.zeros(self.num_configs, dtype=torch.long, device=self.device)
    self.hist_count = torch.zeros(self.num_configs, dtype=torch.long, device=self.device)

    self.env_bin = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)  # config id
    self.env_speed = torch.zeros(env.num_envs, device=self.device)
    self.env_radius = torch.full((env.num_envs,), math.inf, device=self.device)
    self.assigned = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

    env.sgs = self
    print(
      f"[SpeedSGS:{self.mode}] {self.num_bins} bins over {lo:.2f}-{hi:.2f} m/s | "
      f"t={self.target} kappa={self.kappa} T={self.temperature} H={self.history}"
      + (f" | frontier_prob={self.frontier_prob} advance@{self.advance_threshold}"
         if self.mode == "ladder" else "")
      + (f" | {self.num_radii} turn radii {radii[0]:.0f}->{radii[-1]:.1f} m, "
         f"a_lat<={a_lat:.2f} m/s^2, {int(self.feasible.sum())}/{self.num_configs} configs feasible"
         if self.turns else "")
    )

  # ------------------------------------------------------------------ queries
  def speed_of(self, env_ids: torch.Tensor) -> torch.Tensor:
    return self.env_speed[env_ids]

  def radius_of(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Turn radius of the episode (inf = straight: ladder, or no turn axis)."""
    return self.env_radius[env_ids]

  def resample_speed(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Mid-episode speed for env_ids that keeps its radius legal: uniform over the speed
    bins where the episode's radius bin is unlocked and feasible, uniform in the bin."""
    k = self.env_bin[env_ids] % self.num_radii  # radius bin of each env
    ok = (self.radius_frontier.unsqueeze(0) >= k.unsqueeze(1)) & self.feasible[:, k].T  # [n, S]
    ok[ok.sum(dim=1) == 0, 0] = True  # cannot happen (own bin qualifies); keeps multinomial safe
    s = torch.multinomial(ok.float(), 1).squeeze(1)
    lo, hi = self.bin_lo[s], self.bin_hi[s]
    return lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)

  def _unlocked(self) -> torch.Tensor:
    """[S, K] configs that may be sampled in refine (radius at or wider than the frontier)."""
    k = torch.arange(self.num_radii, device=self.device)
    return (k.unsqueeze(0) <= self.radius_frontier.unsqueeze(1)) & self.feasible

  @property
  def v_current(self) -> float:
    return float(self.bin_hi[self.frontier])

  def success_rates(self) -> torch.Tensor:
    return self.hist.sum(-1).float() / self.hist_count.clamp(min=1).float()

  def sgs_probs(self, upto: int) -> torch.Tensor:
    p = self.success_rates()[: upto + 1]
    t, kap, eps = self.target, self.kappa, self.eps
    w = (p + eps) ** (kap * t) * (1.0 - p + eps) ** (kap * (1.0 - t))
    w = torch.clamp(w, min=eps)
    return torch.softmax(torch.log(w + eps) / self.temperature, dim=0)

  # ------------------------------------------------------------------ core
  def _record(self, env_ids: torch.Tensor) -> tuple[int, int]:
    env = self._env
    tm = env.termination_manager
    cmd: GaitCommand = env.command_manager.get_term(self.command_name)
    failed = tm.terminated[env_ids]
    timed_out = tm.time_outs[env_ids] & ~failed
    err_x, yaw, v_cmd, counted_t, yaw_cmd = cmd.episode_outcome(env_ids)
    # Truncated time-outs (randomized initial episode lengths) are not evidence.
    valid = self.assigned[env_ids] & (failed | (timed_out & (counted_t >= self.min_counted_time)))
    if not valid.any():
      return 0, 0
    tol = torch.clamp(self.err_rel * v_cmd, min=self.err_abs)
    tol_yaw = torch.clamp(self.err_yaw_rel * yaw_cmd, min=self.err_yaw)
    success = timed_out & (err_x.abs() <= tol) & (yaw.abs() <= tol_yaw)
    if self.timing_tol is not None:
      tim_err, strides = cmd.timing_outcome(env_ids)
      success &= (strides >= 1) & (tim_err <= self.timing_tol)
    b = self.env_bin[env_ids][valid]
    ok = success[valid]
    # Several envs may write the same bin in one call; sequential writes keep the ring.
    for bin_id in torch.unique(b).tolist():
      outcomes = ok[b == bin_id]
      k = outcomes.numel()
      pos = (self.hist_ptr[bin_id] + torch.arange(k, device=self.device)) % self.history
      self.hist[bin_id, pos[-self.history:]] = outcomes[-self.history:]
      self.hist_ptr[bin_id] = (self.hist_ptr[bin_id] + k) % self.history
      self.hist_count[bin_id] = min(int(self.hist_count[bin_id]) + k, self.history)
    return int(valid.sum()), int(ok.sum())

  def _update_frontier(self) -> None:
    if self.mode != "ladder":
      self._update_radius_frontier()
      return
    f = self.frontier
    rate = float(self.success_rates()[f])
    enough = int(self.hist_count[f]) >= self.min_outcomes
    if enough and rate >= self.advance_threshold and f + 1 < self.num_bins:
      self.frontier = f + 1
    elif enough and self.retreat_threshold > 0 and rate < self.retreat_threshold and f > 0:
      self.frontier = f - 1
    self.frontier_max = max(self.frontier_max, self.frontier)

  def _update_radius_frontier(self) -> None:
    """Refine: each speed bin's radius frontier moves one bin tighter once mastered."""
    if not self.turns:
      return
    s = torch.arange(self.num_bins, device=self.device)
    c = s * self.num_radii + self.radius_frontier
    rate = self.success_rates()[c]
    advance = (
      (self.hist_count[c] >= self.min_outcomes)
      & (rate >= self.advance_threshold)
      & (self.radius_frontier < self.radius_max_bin)
    )
    self.radius_frontier += advance.long()

  def _sample_bins(self, k: int) -> torch.Tensor:
    """Config ids for k new episodes (= speed bins in the ladder)."""
    if self.mode == "refine":
      n_r = self.num_radii
      # Frontier share: uniform speed bin at its radius frontier.
      s = torch.randint(self.num_bins, (k,), device=self.device)
      cfg = s * n_r + self.radius_frontier[s]
      # The rest: SGS (Beta kernel) over every unlocked config.
      sgs = torch.rand(k, device=self.device) >= self.refine_frontier_prob
      n = int(sgs.sum())
      if n:
        p = self.success_rates()
        t, kap, eps = self.target, self.kappa, self.eps
        w = torch.clamp((p + eps) ** (kap * t) * (1.0 - p + eps) ** (kap * (1.0 - t)), min=eps)
        logits = torch.log(w + eps) / self.temperature
        logits[~self._unlocked().flatten()] = -math.inf
        cfg[sgs] = torch.multinomial(torch.softmax(logits, dim=0), n, replacement=True)
      return cfg
    bins = torch.full((k,), self.frontier, dtype=torch.long, device=self.device)
    if self.frontier > 0:
      replay = torch.rand(k, device=self.device) >= self.frontier_prob
      n = int(replay.sum())
      if n:
        bins[replay] = torch.multinomial(self.sgs_probs(self.frontier - 1), n, replacement=True)
    return bins

  def __call__(self, env: ManagerBasedRlEnv, env_ids, **_params) -> dict[str, torch.Tensor]:
    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=self.device)[env_ids]
    n_rec, n_ok = self._record(env_ids)
    self._update_frontier()

    cfg = self._sample_bins(len(env_ids))
    s, k = cfg // self.num_radii, cfg % self.num_radii
    lo, hi = self.bin_lo[s], self.bin_hi[s]
    self.env_bin[env_ids] = cfg
    self.env_speed[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
    self.env_radius[env_ids] = self.radii[k] if self.turns else math.inf
    self.assigned[env_ids] = True

    sr = self.success_rates()
    seen = self.hist_count > 0
    if self.mode == "ladder":
      frontier_success = sr[self.frontier]
    else:  # mean over each speed bin's radius frontier config
      fc = torch.arange(self.num_bins, device=self.device) * self.num_radii + self.radius_frontier
      frontier_success = sr[fc].mean()
    out = {
      "episode_success": torch.tensor(n_ok / max(n_rec, 1)),
      "mean_success_seen": sr[seen].mean() if seen.any() else torch.tensor(0.0),
      "sampled_speed": self.env_speed[env_ids].mean(),
      "frontier_bin": torch.tensor(float(self.frontier)),
      "v_current": torch.tensor(self.v_current),
      "frontier_success": frontier_success,
      "bins_mastered_frac": (sr >= self.advance_threshold)[self.feasible.flatten()].float().mean(),
    }
    if self.turns:
      r = self.env_radius[env_ids]
      out["sampled_radius"] = r.median()
      out["sampled_yaw_rate"] = (self.env_speed[env_ids] / r).mean()
      out["radius_frontier_mean"] = self.radius_frontier.float().mean()
      out["tightest_radius_unlocked"] = self.radii[self.radius_frontier].min()
    return out

  def state_dict(self) -> dict:
    return {
      "frontier": self.frontier,
      "radius_frontier": self.radius_frontier.cpu(),
      "radii": self.radii.cpu(),
      "hist": self.hist.cpu(),
      "hist_ptr": self.hist_ptr.cpu(),
      "hist_count": self.hist_count.cpu(),
    }

  def reset(self, env_ids=None) -> None:
    pass
