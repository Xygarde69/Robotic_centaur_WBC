"""Gamepad steering for play: hold L1 to turn left, R1 to turn right.

Reads the Linux joystick API (/dev/input/js*) directly, no extra packages. L1/R1 are found
through the kernel's button map, so any pad exposing BTN_TL / BTN_TR works (DualSense,
Xbox, ...). The pad's user needs read access to /dev/input (group "input").

The stick position s in [-1, 1] ramps while a shoulder button is held and returns to 0 when
both are released. The command stays on the trained manifold: the commanded curvature is
s * kappa_max(v) with kappa_max = min(1 / r_min, a_lat_max / v^2) (the refine curriculum's
tightest radius and lateral-acceleration cap), and ang_z = v * curvature, as in training.
"""

from __future__ import annotations

import fcntl
import glob
import os
import struct
import sys

import torch

_BTN_TL, _BTN_TR = 0x136, 0x137  # linux/input-event-codes.h
_JSIOCGBUTTONS = 0x80016A12
_JSIOCGBTNMAP = 0x84006A34  # u16[KEY_MAX - BTN_MISC + 1]
_EVENT = struct.Struct("<IhBB")  # time ms, value, type, number
_JS_EVENT_BUTTON, _JS_EVENT_INIT = 0x01, 0x80


def _jsiocgname(n: int) -> int:
  return 0x80006A13 | (n << 16)


def find_pad() -> tuple[str, str]:
  """(device path, name) of the first joystick device that has L1/R1 buttons."""
  for path in sorted(glob.glob("/dev/input/js*")):
    try:
      fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
      continue
    try:
      name = fcntl.ioctl(fd, _jsiocgname(128), b"\0" * 128).split(b"\0")[0].decode(errors="replace")
      if _button_indices(fd) is not None:
        return path, name
    finally:
      os.close(fd)
  raise RuntimeError(
    "no gamepad with L1/R1 found in /dev/input/js* (connected? user in group 'input'?)"
  )


def _button_indices(fd: int) -> tuple[int, int] | None:
  n = fcntl.ioctl(fd, _JSIOCGBUTTONS, b"\0")[0]
  codes = struct.unpack("<512H", fcntl.ioctl(fd, _JSIOCGBTNMAP, b"\0" * 1024))[:n]
  if _BTN_TL in codes and _BTN_TR in codes:
    return codes.index(_BTN_TL), codes.index(_BTN_TR)
  return None


class JoystickTurn:
  """Callable (command, dt) -> None; installed as GaitCommand.joystick."""

  def __init__(self, ramp_time: float = 1.5, release_time: float = 0.5):
    self.path, self.name = find_pad()
    self._fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
    self._left, self._right = _button_indices(self._fd)  # type: ignore[misc]
    self._held = [False, False]  # L1, R1
    self.ramp_time = ramp_time  # s from 0 to full lock while held
    self.release_time = release_time  # s from full lock back to 0 after release
    self.s = 0.0
    self._since_print = 0.0
    print(f"[INFO]: Joystick {self.name} ({self.path}): L1 = turn left, R1 = turn right")

  def _poll(self) -> None:
    while True:
      try:
        data = os.read(self._fd, _EVENT.size * 32)
      except BlockingIOError:
        return
      except OSError:  # pad unplugged: stop turning
        self._held = [False, False]
        return
      for i in range(0, len(data) - _EVENT.size + 1, _EVENT.size):
        _, value, kind, number = _EVENT.unpack_from(data, i)
        if kind & ~_JS_EVENT_INIT == _JS_EVENT_BUTTON:
          if number == self._left:
            self._held[0] = bool(value)
          elif number == self._right:
            self._held[1] = bool(value)

  def __call__(self, command, dt: float) -> None:
    self._poll()
    left, right = self._held
    if left != right:
      self.s += (1.0 if left else -1.0) * dt / self.ramp_time
    elif not left:
      step = dt / self.release_time
      self.s = max(0.0, self.s - step) if self.s > 0 else min(0.0, self.s + step)
    self.s = max(-1.0, min(1.0, self.s))
    command.set_stick_turn(self.s)
    self._since_print += dt
    if self._since_print >= 0.5 and (left or right or self.s != 0.0):
      self._since_print = 0.0
      v = float(command.speed[0])
      w = float(command.yaw_rate[0])
      r = "straight" if abs(w) < 1e-3 else f"radius {v / abs(w):.1f} m"
      print(f"\r[turn] stick {self.s:+.2f}  yaw rate {w:+.2f} rad/s  {r}        ", end="", file=sys.stdout, flush=True)

  def close(self) -> None:
    os.close(self._fd)


def attach(env) -> JoystickTurn:
  """Install a JoystickTurn on the env's gait command (refine tasks only)."""
  term = env.unwrapped.command_manager.get_term("gait")
  if not term.cfg.turn_radii:
    raise ValueError("--joystick needs a Refine task (the only phase trained with turns)")
  pad = JoystickTurn()
  term.joystick = pad
  return pad
