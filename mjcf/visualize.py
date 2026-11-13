"""Simple interactive viewer for the Centaur MuJoCo model.

Install the dependency with:
    python -m pip install mujoco

Run with:
    python visualize.py
    python visualize.py --simulate
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).with_name("centaur.xml")


def main() -> None:
    parser = argparse.ArgumentParser(description="View the Centaur MJCF model")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="run the physics simulation instead of displaying a static pose",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Start from the pose stored in the XML. Keep the static view upright and
    # still; --simulate enables gravity and dynamics.
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        while viewer.is_running():
            if args.simulate:
                step_start = time.perf_counter()
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - step_start)))
            else:
                viewer.sync()
                time.sleep(1 / 60)


if __name__ == "__main__":
    main()
