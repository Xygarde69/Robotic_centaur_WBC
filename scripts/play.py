"""Script to play RL agent with RSL-RL."""

import sys as _sys
from pathlib import Path as _Path

# Make the repo root importable (centaur_rl) without needing `pip install -e .`.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  speed: float | None = None
  """Multi-gait tasks: hold the commanded forward speed (m/s) for the whole episode
  instead of resampling it across the gait's range. For a LADDER checkpoint the default
  is the fastest speed it had mastered when it was saved (its SGS frontier)."""
  radius: float | None = None
  """Multi-gait tasks: hold a turn of this radius (m, + = left, - = right, 0 = straight)
  for the whole episode: ang_z = lin_x_cmd / radius. Default: refine tasks pick a random
  feasible radius per episode, the other phases drive straight."""

  joystick: bool = False
  """Multi-gait Refine tasks: steer with a gamepad. Hold L1 to turn left and R1 to turn
  right; the turn tightens while held and straightens on release, within the limits the
  curriculum trains (ang_z = v * curvature). Combine with --speed to set the speed."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  # Constant command speed: --speed, or for a multi-gait ladder checkpoint the fastest
  # speed it had mastered (its SGS frontier at save time).
  play_speed = cfg.speed
  if "gait" in env_cfg.commands and play_speed is None and resume_path is not None:
    from centaur_rl.tasks.Multi_Policy.pipeline import ladder_play_speed

    infos = torch.load(resume_path, map_location="cpu", weights_only=False).get("infos")
    found = ladder_play_speed(task_id, infos)
    if found is not None:
      play_speed, note = found
      print(f"[INFO]: {note}")
  if play_speed is not None:
    if "gait" not in env_cfg.commands:
      raise ValueError("--speed needs a multi-gait task (the 'gait' command)")
    gait_cmd = env_cfg.commands["gait"]
    gait_cmd.speed_range = (play_speed, play_speed)
    gait_cmd.resample_mid_episode = False
    gait_cmd.resampling_time_range = (1e6, 1e6)
    print(f"[INFO]: Constant command speed {play_speed:.2f} m/s")
  if cfg.radius is not None:
    if "gait" not in env_cfg.commands:
      raise ValueError("--radius needs a multi-gait task (the 'gait' command)")
    env_cfg.commands["gait"].play_radius = cfg.radius
    side = "straight" if cfg.radius == 0 else ("left" if cfg.radius > 0 else "right")
    print(f"[INFO]: Constant turn radius {abs(cfg.radius):.2f} m ({side})")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  if cfg.joystick:
    if "gait" not in env_cfg.commands:
      raise ValueError("--joystick needs a multi-gait task (the 'gait' command)")
    from centaur_rl.tasks.Multi_Policy.tools.joystick import attach

    attach(env)

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import centaur_rl.tasks  # noqa: F401  (registers Centaur-* tasks)

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  # mjlab turns off bare-flag booleans; accept a bare `--joystick` anyway.
  for i, a in enumerate(remaining_args):
    if a == "--joystick" and (i + 1 == len(remaining_args) or remaining_args[i + 1].startswith("--")):
      remaining_args[i] = "--joystick=True"
  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
