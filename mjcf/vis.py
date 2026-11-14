"""Visualise centaur_train.xml and hold the default stance pose.

Pose matches INIT_STATE in centaur_rl/assets/robots/centaur/centaur_constants.py.

    python vis.py            # kinematic: pose is re-applied every frame (static)
    python vis.py --physics  # simulate, with a PD controller holding the stance
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = Path(__file__).with_name("centaur_train.xml")

KNEE_BEND = 0.5235988  # 30 deg
KNEE_FLEX_SIGN = {"FL": 1.0, "FR": -1.0, "RL": -1.0, "RR": 1.0}
BASE_POS = (0.0, 0.0, 0.60)  # feet just touch the ground in this stance

KP, KD = 150.0, 5.0  # same gains as the training actuators


def stance_qpos(model):
  qpos = model.qpos0.copy()
  qpos[0:3] = BASE_POS
  qpos[3:7] = (1, 0, 0, 0)
  for leg, s in KNEE_FLEX_SIGN.items():
    for suffix, val in (("_1", 0.0), ("_2", -0.5 * KNEE_BEND * s), ("_3", KNEE_BEND * s)):
      qpos[model.joint(leg + suffix).qposadr[0]] = val
  return qpos


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--physics", action="store_true", help="simulate with PD hold")
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
  data = mujoco.MjData(model)
  q_stance = stance_qpos(model)
  data.qpos[:] = q_stance
  mujoco.mj_forward(model, data)

  leg_qadr = model.jnt_qposadr[1:]
  leg_dadr = model.jnt_dofadr[1:]

  with mujoco.viewer.launch_passive(model, data) as viewer:
    # Show the (group 2) visual meshes; collision geoms are group 3.
    viewer.opt.geomgroup[2] = 1
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20

    while viewer.is_running():
      t0 = time.perf_counter()
      if args.physics:
        err = q_stance[leg_qadr] - data.qpos[leg_qadr]
        data.qfrc_applied[:] = 0
        data.qfrc_applied[leg_dadr] = KP * err - KD * data.qvel[leg_dadr]
        mujoco.mj_step(model, data)
        time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - t0)))
      else:
        data.qpos[:] = q_stance
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        time.sleep(1 / 60)
      viewer.sync()


if __name__ == "__main__":
  main()
