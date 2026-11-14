"""Speed SGS for one gait, straight line on flat ground.

Success-Guided Sampling (arXiv:2610.12465) over a single axis: the forward speed,
split into `num_bins` bins over the gait's SGS range. Each episode draws its speed
uniformly inside its bin. No terrain tiles, no turn axis (straight line only).

Two modes:

  ladder  V_current = the frontier bin. A share `frontier_prob` of the episodes is
          sampled AT the frontier; the rest is SGS-weighted (Beta kernel) over the
          bins already unlocked (replay of slower speeds). The frontier advances when
          its success rate over its last `history` outcomes reaches
          `advance_threshold` (with at least `min_outcomes` of them), and retreats if
          it falls below `retreat_threshold` after advancing (optional).
          => speeds climb from v_min to v_max as each speed is mastered, instead of on
          a fixed iteration schedule.

  refine  every bin is unlocked; plain SGS over the whole range. The command term
          additionally resamples the speed mid-episode (up and down).

Success of an episode (decided at its reset):
  ran to time-out (no failure termination), and over the counted steps (after
  settle_time) |mean(v_x - v_cmd)| <= max(err_abs, err_rel * mean|v_cmd|) and
  |mean(yaw rate)| <= err_yaw. With timing_tol set (walk) the footfalls must also keep the
  clock's spacing: at least one stride, and the mean over strides of the worst deviation
  of the left/right and hind->fore touchdown spacings <= timing_tol (cycle fractions). Mean of the SIGNED error: the in-stride velocity ripple
  of the flight gaits is not a tracking error.
"""

from __future__ import annotations

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

    # Ladder.
    self.frontier_prob = p.get("frontier_prob", 0.8)
    self.advance_threshold = p.get("advance_threshold", 0.8)
    self.retreat_threshold = p.get("retreat_threshold", 0.0)  # 0 disables retreating
    self.min_outcomes = int(p.get("min_outcomes", 50))
    self.frontier = 0 if self.mode == "ladder" else self.num_bins - 1
    self.frontier_max = self.frontier

    self.hist = torch.zeros(self.num_bins, self.history, dtype=torch.bool, device=self.device)
    self.hist_ptr = torch.zeros(self.num_bins, dtype=torch.long, device=self.device)
    self.hist_count = torch.zeros(self.num_bins, dtype=torch.long, device=self.device)

    self.env_bin = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
    self.env_speed = torch.zeros(env.num_envs, device=self.device)
    self.assigned = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

    env.sgs = self
    print(
      f"[SpeedSGS:{self.mode}] {self.num_bins} bins over {lo:.2f}-{hi:.2f} m/s | "
      f"t={self.target} kappa={self.kappa} T={self.temperature} H={self.history}"
      + (f" | frontier_prob={self.frontier_prob} advance@{self.advance_threshold}"
         if self.mode == "ladder" else "")
    )

  # ------------------------------------------------------------------ queries
  def speed_of(self, env_ids: torch.Tensor) -> torch.Tensor:
    return self.env_speed[env_ids]

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
    err_x, yaw, v_cmd, counted_t = cmd.episode_outcome(env_ids)
    # Truncated time-outs (randomized initial episode lengths) are not evidence.
    valid = self.assigned[env_ids] & (failed | (timed_out & (counted_t >= self.min_counted_time)))
    if not valid.any():
      return 0, 0
    tol = torch.clamp(self.err_rel * v_cmd, min=self.err_abs)
    success = timed_out & (err_x.abs() <= tol) & (yaw.abs() <= self.err_yaw)
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
      return
    f = self.frontier
    rate = float(self.success_rates()[f])
    enough = int(self.hist_count[f]) >= self.min_outcomes
    if enough and rate >= self.advance_threshold and f + 1 < self.num_bins:
      self.frontier = f + 1
    elif enough and self.retreat_threshold > 0 and rate < self.retreat_threshold and f > 0:
      self.frontier = f - 1
    self.frontier_max = max(self.frontier_max, self.frontier)

  def _sample_bins(self, k: int) -> torch.Tensor:
    if self.mode == "refine":
      return torch.multinomial(self.sgs_probs(self.num_bins - 1), k, replacement=True)
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

    bins = self._sample_bins(len(env_ids))
    lo, hi = self.bin_lo[bins], self.bin_hi[bins]
    self.env_bin[env_ids] = bins
    self.env_speed[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
    self.assigned[env_ids] = True

    sr = self.success_rates()
    seen = self.hist_count > 0
    out = {
      "episode_success": torch.tensor(n_ok / max(n_rec, 1)),
      "mean_success_seen": sr[seen].mean() if seen.any() else torch.tensor(0.0),
      "sampled_speed": self.env_speed[env_ids].mean(),
      "frontier_bin": torch.tensor(float(self.frontier)),
      "v_current": torch.tensor(self.v_current),
      "frontier_success": sr[self.frontier],
      "bins_mastered_frac": (sr >= self.advance_threshold).float().mean(),
    }
    return out

  def state_dict(self) -> dict:
    return {
      "frontier": self.frontier,
      "hist": self.hist.cpu(),
      "hist_ptr": self.hist_ptr.cpu(),
      "hist_count": self.hist_count.cpu(),
    }

  def reset(self, env_ids=None) -> None:
    pass
