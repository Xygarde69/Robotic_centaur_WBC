"""Gamepad state over the Linux joystick API (/dev/input/js*), no extra packages.

Buttons and axes are looked up by their kernel codes (linux/input-event-codes.h), not by
index, so the mapping holds for any pad the kernel maps the standard way (DualSense via
hid-playstation, DualShock 4, Xbox, ...). The user needs read access to /dev/input
(group "input").

  buttons  pressed(code) while held, presses(code) = presses since the last call
  axes     axis(code) in [-1, 1]; trigger(code) in [0, 1] (released = 0)
"""

from __future__ import annotations

import fcntl
import glob
import os
import struct

# linux/input-event-codes.h
BTN_SOUTH, BTN_EAST, BTN_NORTH, BTN_WEST = 0x130, 0x131, 0x133, 0x134  # cross, circle, triangle, square
BTN_TL, BTN_TR, BTN_TL2, BTN_TR2 = 0x136, 0x137, 0x138, 0x139  # L1, R1, L2 (click), R2 (click)
ABS_X, ABS_Y, ABS_Z, ABS_RX, ABS_RY, ABS_RZ = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05  # Z = L2, RZ = R2

_JSIOCGAXES, _JSIOCGBUTTONS = 0x80016A11, 0x80016A12
_JSIOCGAXMAP, _JSIOCGBTNMAP = 0x80406A32, 0x84006A34
_EVENT = struct.Struct("<IhBB")  # time ms, value, type, number
_JS_EVENT_BUTTON, _JS_EVENT_AXIS, _JS_EVENT_INIT = 0x01, 0x02, 0x80


def _jsiocgname(n: int) -> int:
  return 0x80006A13 | (n << 16)


def _maps(fd: int) -> tuple[list[int], list[int]]:
  nb = fcntl.ioctl(fd, _JSIOCGBUTTONS, b"\0")[0]
  na = fcntl.ioctl(fd, _JSIOCGAXES, b"\0")[0]
  buttons = list(struct.unpack("<512H", fcntl.ioctl(fd, _JSIOCGBTNMAP, b"\0" * 1024))[:nb])
  axes = list(fcntl.ioctl(fd, _JSIOCGAXMAP, b"\0" * 64))[:na]
  return buttons, axes


def find_pad(required_buttons=(BTN_TL, BTN_TR), required_axes=()) -> tuple[str, str]:
  """(device path, name) of the first joystick that has the required buttons and axes.
  Skips the DualSense's separate motion-sensor device (no buttons)."""
  for path in sorted(glob.glob("/dev/input/js*")):
    try:
      fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
      continue
    try:
      name = fcntl.ioctl(fd, _jsiocgname(128), b"\0" * 128).split(b"\0")[0].decode(errors="replace")
      buttons, axes = _maps(fd)
      if all(b in buttons for b in required_buttons) and all(a in axes for a in required_axes):
        return path, name
    finally:
      os.close(fd)
  raise RuntimeError(
    "no matching gamepad in /dev/input/js* (connected? user in group 'input'?)"
  )


class Gamepad:
  def __init__(self, required_buttons=(BTN_TL, BTN_TR), required_axes=()):
    self.path, self.name = find_pad(required_buttons, required_axes)
    self._fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
    self._btn_codes, self._ax_codes = _maps(self._fd)
    self._held: dict[int, bool] = {}
    self._presses: dict[int, int] = {}
    self._axis: dict[int, float] = {}
    self.connected = True

  def poll(self) -> None:
    while True:
      try:
        data = os.read(self._fd, _EVENT.size * 64)
      except BlockingIOError:
        return
      except OSError:  # unplugged: release everything
        self._held.clear()
        self._axis.clear()
        self.connected = False
        return
      for i in range(0, len(data) - _EVENT.size + 1, _EVENT.size):
        _, value, kind, number = _EVENT.unpack_from(data, i)
        init = bool(kind & _JS_EVENT_INIT)
        kind &= ~_JS_EVENT_INIT
        if kind == _JS_EVENT_BUTTON and number < len(self._btn_codes):
          code = self._btn_codes[number]
          if value and not init and not self._held.get(code, False):
            self._presses[code] = self._presses.get(code, 0) + 1
          self._held[code] = bool(value)
        elif kind == _JS_EVENT_AXIS and number < len(self._ax_codes):
          self._axis[self._ax_codes[number]] = value / 32767.0

  def pressed(self, code: int) -> bool:
    return self._held.get(code, False)

  def presses(self, code: int) -> int:
    """Presses of a button since the last call (so quick taps between polls count)."""
    return self._presses.pop(code, 0)

  def axis(self, code: int) -> float:
    return self._axis.get(code, 0.0)

  def trigger(self, code: int) -> float:
    """Analog trigger in [0, 1]: the kernel reports -1 released, +1 fully pressed. Before
    the first event from it a trigger reads 0."""
    if code not in self._axis:
      return 0.0
    return min(max((self._axis[code] + 1.0) / 2.0, 0.0), 1.0)

  def close(self) -> None:
    os.close(self._fd)
