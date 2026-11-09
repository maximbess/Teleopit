"""Native pauseable viewer and small video export for the coupled trial."""

import time
from contextlib import ExitStack
from threading import Event

import mujoco
import mujoco.viewer

from .run import run_trial


def camera(cam):
    cam.lookat[:] = (-0.6, 0, 1.4)
    cam.distance = 3.2
    cam.azimuth = 150
    cam.elevation = -15


def show(config, output):
    paused = Event()

    def key(code):
        if code == 32:
            paused.clear() if paused.is_set() else paused.set()

    with ExitStack() as stack:
        window = None

        def frame(env, row):
            nonlocal window
            if window is None:
                window = stack.enter_context(
                    mujoco.viewer.launch_passive(env.model, env.data, key_callback=key)
                )
                camera(window.cam)
                window.opt.geomgroup[3] = 0
                window.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
                window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
                window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = False
            start = time.monotonic()
            while window.is_running():
                window.sync()
                if not paused.is_set():
                    break
                time.sleep(config.control_dt)
            if not window.is_running():
                env.abort(["viewer_closed"])
            time.sleep(max(0, config.control_dt - (time.monotonic() - start)))

        return run_trial(config, output, frame)


def record(config, output):
    from pathlib import Path

    import imageio.v2 as imageio

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    renderer = None
    counter = 0
    cam = mujoco.MjvCamera()
    camera(cam)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    with imageio.get_writer(str(output / "trial.mp4"), fps=10) as writer:

        def frame(env, row):
            nonlocal renderer, counter
            if renderer is None:
                renderer = mujoco.Renderer(env.model, height=480, width=640)
            if counter % 5 == 0:
                renderer.update_scene(env.data, camera=cam, scene_option=opt)
                pixels = renderer.render()
                pixels = label_frame(pixels, env, row)
                writer.append_data(pixels)
                if counter == 0:
                    imageio.imwrite(output / "initial.png", pixels)
                imageio.imwrite(output / "last.png", pixels)
            counter += 1

        try:
            return run_trial(config, output, frame)
        finally:
            if renderer is not None:
                renderer.close()


def label_frame(pixels, env, row):
    import numpy as np
    from PIL import Image, ImageDraw

    picture = Image.fromarray(pixels)
    draw = ImageDraw.Draw(picture)
    draw.rectangle((0, 0, picture.width, 44), fill="black")
    draw.text(
        (8, 5),
        f"t={row['time']:.2f}s  {env.phase}  assistance={row['assistance_gain']:.2f}",
        fill="white",
    )
    status = "passed" if env.hold_passed else "NOT passed"
    draw.text(
        (8, 24), f"Physical feasibility test: static hold {status}", fill="orange"
    )
    return np.array(picture)
