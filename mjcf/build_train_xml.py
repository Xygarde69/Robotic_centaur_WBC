"""Generate the legged-locomotion training MJCF (`centaur_train.xml`) from `centaur.xml`.

What this does to the CAD-exported model:

* Upper body (torso, neck, both 6-DoF arms), controlled by `UPPER_BODY_MODE`:
    "remove" (current default): the whole Torso_1 subtree and its meshes are deleted,
              leaving a plain quadruped (~35.6 kg).
    "weld":   kept rigidly attached (joints/actuators removed, geometry + mass kept, ~58 kg),
              frozen at `UPPER_BODY_POSE`.
* The floating base is re-oriented so the free-joint body `base_link` is x-forward,
  y-left, z-up (what mjlab's velocity commands / projected gravity assume). In the CAD
  export base_link is +y-up / +z-forward (torso end = front).
* Collision: ONLY the four calf meshes collide (`<leg>_calf_collision`, convex hull of
  the calf mesh). Every other geom (base, hips, thighs, upper body) is visual only.
* Each calf gets a foot site (`<leg>`) near the calf tip, used for foot height/slip terms.
* An `imu` site plus the sensors the velocity task expects are added
  (`imu_ang_vel`, `imu_lin_vel`, `imu_lin_acc`, `root_angmom`).
* No actuators: mjlab adds them from `centaur_constants.py`.

Run:  python mjcf/build_train_xml.py      (re-run after changing centaur.xml)
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "centaur.xml"
OUT = HERE / "centaur_train.xml"

LEGS = ("FR", "FL", "RR", "RL")
UPPER_BODY_ROOT = "Torso_1"

# "remove" -> quadruped only; "weld" -> static upper-body payload (see module docstring).
UPPER_BODY_MODE = "remove"

# Joint angles (rad) the upper body is frozen at in "weld" mode. Zero = CAD pose (T-pose).
UPPER_BODY_POSE: dict[str, float] = {}

# Rotation taking old base_link axes to the new x-fwd/z-up base: old z(front)->x,
# old x->y, old y(up)->z. Cyclic permutation == 120 deg about (1,1,1).
Q_BASE = np.array([0.5, 0.5, 0.5, 0.5])

# Foot site in calf (<leg>_3) frame: tip of the calf mesh is at y ~= -0.32; the site
# sits 2 cm above it (like the center of Go2's foot sphere).
FOOT_SITE_POS = (0.0, -0.30, 0.001)
FOOT_SITE_SIZE = 0.02


# ----------------------------------------------------------------------------- math
def quat_mul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ])


def quat_to_mat(q):
  w, x, y, z = q
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])


def axis_angle(axis, angle):
  axis = np.asarray(axis, float)
  axis = axis / np.linalg.norm(axis)
  return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def vec(s: str | None, default):
  return np.array([float(v) for v in s.split()]) if s else np.array(default, float)


def fmt(v) -> str:
  return " ".join(f"{x:.9g}" for x in np.asarray(v).ravel())


# ----------------------------------------------------------------------------- xml
def inline_includes(elem: ET.Element, base: Path) -> None:
  for i, child in enumerate(list(elem)):
    if child.tag == "include":
      inc = ET.parse(base / child.get("file")).getroot()
      elem.remove(child)
      for j, sub in enumerate(list(inc)):
        elem.insert(i + j, sub)
      inline_includes(elem, base)  # indices shifted; restart
      return
    inline_includes(child, base)


def find_body(root: ET.Element, name: str) -> ET.Element:
  b = root.find(f".//body[@name='{name}']")
  assert b is not None, name
  return b


def weld_upper_body(body: ET.Element) -> None:
  """Remove joints and apply the frozen pose (recursively)."""
  for j in body.findall("joint"):
    angle = UPPER_BODY_POSE.get(j.get("name"), 0.0)
    if angle:
      q = vec(body.get("quat"), (1, 0, 0, 0))
      body.set("quat", fmt(quat_mul(q, axis_angle(vec(j.get("axis"), (0, 0, 1)), angle))))
    body.remove(j)
  for g in body.findall("geom"):
    if g.get("class") == "collision":
      body.remove(g)
  for child in body.findall("body"):
    weld_upper_body(child)


def strip_collisions(elem: ET.Element, keep: set[str]) -> None:
  """Remove collision geoms from every body except those named in `keep` (recursive)."""
  for body in elem.iter("body"):
    if body.get("name") in keep:
      continue
    for g in body.findall("geom"):
      if g.get("class") == "collision":
        body.remove(g)


def build() -> ET.ElementTree:
  tree = ET.parse(SRC)
  root = tree.getroot()
  inline_includes(root, SRC.parent)

  root.set("model", "centaur_legs")
  compiler = root.find("compiler")
  compiler.set("fusestatic", "false")

  # Drop actuators/sensors/keyframes/scene; mjlab owns those.
  for tag in ("actuator", "sensor", "keyframe"):
    for e in root.findall(tag):
      root.remove(e)
  asset = root.find("asset")
  for e in list(asset):
    if e.tag in ("texture", "material"):
      asset.remove(e)
  defaults = root.find("default")
  defaults.find("joint").attrib = {"damping": "0.1"}
  coll = defaults.find("default[@class='collision']/geom")
  coll.attrib = {"group": "3", "contype": "1", "conaffinity": "0", "condim": "1"}
  vis = defaults.find("default[@class='visual']/geom")
  vis.set("rgba", "0.894 0.894 0.894 1")
  root.find("option").attrib = {"timestep": "0.005"}

  wb = root.find("worldbody")
  old_base = find_body(wb, "base_link")

  # ---- new free-floating base_link, x-forward / z-up
  new_wb = ET.Element("worldbody")
  base = ET.SubElement(new_wb, "body", name="base_link", pos="0 0 0.6")
  ET.SubElement(base, "freejoint", name="floating_base_joint")
  R = quat_to_mat(Q_BASE)

  for child in list(old_base):
    c = copy.deepcopy(child)
    if c.tag == "inertial":
      p = vec(c.get("pos"), (0, 0, 0))
      ixx, iyy, izz, ixy, ixz, iyz = vec(c.get("fullinertia"), (0,) * 6)
      I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
      I2 = R @ I @ R.T
      c.set("pos", fmt(R @ p))
      c.set("fullinertia", fmt([I2[0, 0], I2[1, 1], I2[2, 2], I2[0, 1], I2[0, 2], I2[1, 2]]))
    elif c.tag == "geom":
      c.set("quat", fmt(Q_BASE))
    elif c.tag == "body":
      c.set("pos", fmt(R @ vec(c.get("pos"), (0, 0, 0))))
      c.set("quat", fmt(quat_mul(Q_BASE, vec(c.get("quat"), (1, 0, 0, 0)))))
    elif c.tag == "site":
      continue  # replaced by imu site below
    base.append(c)
  ET.SubElement(base, "site", name="imu", pos="0 0 0", size="0.01")

  # ---- upper body
  upper = find_body(base, UPPER_BODY_ROOT)
  if UPPER_BODY_MODE == "remove":
    base.remove(upper)
    used = {g.get("mesh") for g in new_wb.iter("geom") if g.get("mesh")}
    for m in asset.findall("mesh"):
      if m.get("name") not in used:
        asset.remove(m)
  elif UPPER_BODY_MODE == "weld":
    weld_upper_body(upper)
  else:
    raise ValueError(f"UPPER_BODY_MODE must be 'remove' or 'weld', got {UPPER_BODY_MODE!r}")

  # ---- collisions: calf meshes only
  calves = {f"{leg}_3" for leg in LEGS}
  strip_collisions(base, keep=calves)
  for leg in LEGS:
    calf = find_body(base, f"{leg}_3")
    coll = [g for g in calf.findall("geom") if g.get("class") == "collision"]
    assert len(coll) == 1, leg
    coll[0].set("name", f"{leg}_calf_collision")
    ET.SubElement(calf, "site", {"name": leg, "pos": fmt(FOOT_SITE_POS),
                                 "size": str(FOOT_SITE_SIZE)})

  root.remove(wb)
  root.insert(list(root).index(asset) + 1, new_wb)

  sensor = ET.SubElement(root, "sensor")
  ET.SubElement(sensor, "gyro", name="imu_ang_vel", site="imu")
  ET.SubElement(sensor, "velocimeter", name="imu_lin_vel", site="imu")
  ET.SubElement(sensor, "accelerometer", name="imu_lin_acc", site="imu")
  ET.SubElement(sensor, "subtreeangmom", name="root_angmom", body="base_link")

  ET.indent(tree, "  ")
  return tree


if __name__ == "__main__":
  tree = build()
  with open(OUT, "w") as f:
    f.write("<!-- AUTO-GENERATED by build_train_xml.py from centaur.xml. Do not edit. -->\n")
    tree.write(f, encoding="unicode")
  print(f"wrote {OUT}")
