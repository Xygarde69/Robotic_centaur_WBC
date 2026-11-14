"""Runner for multi-gait stages: how a stage inherits from its parent checkpoint.

fork      (new gait)  actor weights + actor obs normalizer from the parent; critic and
                      optimizer fresh; actor frozen for critic_warmup_iterations; action
                      std raised to std_floor; actor normalizer count capped so it
                      re-adapts (and it keeps updating throughout, never frozen).
continue  (same gait) actor, critic and optimizer from the parent.

Also: loads handoff banks into the reset event, exports ONNX on every save and writes a
result JSON at the end (the orchestrator chains stages through it).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner


class MultiGaitRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    self.stage: dict = dict(train_cfg.pop("stage", None) or {})
    super().__init__(env, train_cfg, log_dir, device)
    self._log_dir = log_dir
    self._updates = 0
    self._actor_frozen = False
    self._warmup = 0
    self._wrap_update()

  # ------------------------------------------------------------------ stage init
  def _apply_stage_init(self) -> None:
    mode = self.stage.get("init_mode", "none")
    ckpt = self.stage.get("init_checkpoint", "")
    if mode != "none":
      if not ckpt:
        raise ValueError(f"stage init_mode={mode} needs --agent.stage.init-checkpoint")
      loaded = torch.load(ckpt, map_location=self.device, weights_only=False)
      if mode == "fork":
        self.alg.load(loaded, {"actor": True}, strict=True)
        self._raise_std(float(self.stage.get("std_floor", 0.0)))
        self._cap_normalizer(int(self.stage.get("normalizer_window", 0)))
        self._warmup = int(self.stage.get("critic_warmup_iterations", 0))
      elif mode == "continue":
        self.alg.load(loaded, {"actor": True, "critic": True, "optimizer": True}, strict=True)
      else:
        raise ValueError(f"unknown init_mode {mode}")
      print(f"[MultiGait] {self.stage.get('name', '')}: {mode} from {ckpt}")

    banks = [b for b in str(self.stage.get("handoff_banks", "") or "").split(",") if b]
    if banks:
      prob = float(self.stage.get("handoff_prob", -1.0))
      self.env.unwrapped.gait_reset.load_banks(banks, None if prob < 0 else prob)

  def _raise_std(self, floor: float) -> None:
    if floor <= 0:
      return
    dist = getattr(self.alg.actor, "distribution", None)
    with torch.no_grad():
      if hasattr(dist, "std_param"):
        dist.std_param.clamp_(min=floor)
      elif hasattr(dist, "log_std_param"):
        dist.log_std_param.clamp_(min=float(torch.log(torch.tensor(floor))))
      else:
        print("[MultiGait] WARNING: no std parameter found; std floor not applied")
        return
    print(f"[MultiGait] action std floor {floor}")

  def _cap_normalizer(self, window: int) -> None:
    norm = getattr(self.alg.actor, "obs_normalizer", None)
    if window <= 0 or norm is None or not hasattr(norm, "count"):
      return
    with torch.no_grad():
      norm.count.clamp_(max=window)
    print(f"[MultiGait] actor obs normalizer count capped at {window}")

  # ------------------------------------------------------------------ critic warmup
  def _set_actor_trainable(self, trainable: bool) -> None:
    for p in self.alg.actor.parameters():
      p.requires_grad_(trainable)
    self._actor_frozen = not trainable

  def _wrap_update(self) -> None:
    update = self.alg.update

    def update_with_warmup() -> dict[str, float]:
      frozen = self._updates < self._warmup
      if frozen != self._actor_frozen:
        self._set_actor_trainable(not frozen)
        print(f"[MultiGait] actor {'FROZEN (critic warmup)' if frozen else 'unfrozen'} "
              f"at update {self._updates}")
      out = update()
      out["actor_frozen"] = float(frozen)
      self._updates += 1
      return out

    self.alg.update = update_with_warmup  # type: ignore[method-assign]

  # ------------------------------------------------------------------ learn / save
  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    resumed = self.current_learning_iteration > 0
    if resumed:
      print("[MultiGait] resumed run: stage init skipped")
    else:
      self._apply_stage_init()
    super().learn(num_learning_iterations, init_at_random_ep_len)
    self._write_result()

  def save(self, path: str, infos=None) -> None:
    sgs = getattr(self.env.unwrapped, "sgs", None)
    infos = {**(infos or {}), "stage": self.stage, "sgs": sgs.state_dict() if sgs else None}
    super().save(path, infos)
    policy_dir = os.path.dirname(path)
    try:
      self.export_policy_to_onnx(policy_dir, "policy.onnx")
      attach_metadata_to_onnx(
        os.path.join(policy_dir, "policy.onnx"), get_base_metadata(self.env.unwrapped, "local")
      )
    except Exception as e:  # export must never kill a long run
      print(f"[MultiGait] ONNX export failed: {e}")

  def _write_result(self) -> None:
    out = self.stage.get("result_file", "")
    if not out or self.logger.writer is None:
      return
    log_dir = Path(self.logger.log_dir)
    ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    sgs = getattr(self.env.unwrapped, "sgs", None)
    result = {
      "stage": self.stage.get("name", ""),
      "log_dir": str(log_dir.resolve()),
      "final_checkpoint": str(ckpts[-1].resolve()) if ckpts else "",
      "checkpoints": {int(p.stem.split("_")[1]): str(p.resolve()) for p in ckpts},
      "iterations": self.current_learning_iteration + 1,
      "v_current": sgs.v_current if sgs else None,
      "frontier_bin": sgs.frontier if sgs else None,
      "num_bins": sgs.num_bins if sgs else None,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"[MultiGait] result -> {out}")
