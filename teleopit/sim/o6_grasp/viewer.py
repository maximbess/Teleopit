"""Pausable MuJoCo viewer with optional contact arrows and a support indicator."""

import time
from threading import Event

import mujoco
import mujoco.viewer
import numpy as np

from .env import O6GraspEnv


def camera_setup(camera, config=None):
    camera.lookat[:] = (0.02, 0, 0.60)
    camera.distance = 0.42
    # View the broad ladder rung from the hand side instead of hiding the palm.
    camera.azimuth = (
        315 if config is not None and config.cross_section == "rectangular" else 135
    )
    camera.elevation = -15


def show(config):
    env = O6GraspEnv(config)
    config = env.config
    _, info = env.reset()
    paused = Event()
    done = False

    def on_key(keycode):
        # The UI thread is the sole writer; the simulation thread only reads.
        if keycode == 32:  # Space
            if paused.is_set():
                paused.clear()
            else:
                paused.set()

    with mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=on_key
    ) as viewer:
        camera_setup(viewer.cam, config)
        viewer.opt.geomgroup[3] = 0
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        while viewer.is_running():
            started = time.monotonic()
            is_paused = paused.is_set()
            if not is_paused:
                _, _, _, done, info = env.step(np.zeros(6))
            with viewer.lock():
                scene = viewer.user_scn
                scene.ngeom = 1
                supported = info["fixture_gain"] > 0
                color = [1.0, 0.55, 0.0, 1.0] if supported else [0.0, 0.8, 0.2, 1.0]
                mujoco.mjv_initGeom(
                    scene.geoms[0],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.006] * 3),
                    env.data.site("wrist_mount").xpos,
                    np.eye(3).flatten(),
                    np.array(color, dtype=np.float32),
                )
                prefix = "PAUSED | " if is_paused else ""
                scene.geoms[
                    0
                ].label = f"{prefix}{info['mode']} fixture={info['fixture_gain']:.2f} {info['verification']}"
            viewer.sync()
            if done:
                break
            time.sleep(max(0.0, config.control_dt - (time.monotonic() - started)))


def render_frames(config, output):
    """Export inspectable OPEN/CLOSE/unsupported/RELEASE views without launching a GUI."""
    from pathlib import Path

    import imageio.v3 as imageio

    env = O6GraspEnv(config)
    config = env.config
    camera = mujoco.MjvCamera()
    camera_setup(camera, config)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    times = [
        0,
        sum(config.phase_durations[:2]),
        sum(config.phase_durations[:2])
        + config.fixture_ramp
        + (config.phase_durations[2] - config.fixture_ramp - config.unload_duration)
        / 2,
        sum(config.phase_durations) - config.control_dt,
    ]
    with mujoco.Renderer(env.model, height=480, width=640) as renderer:
        for name, target in zip(("open", "closed", "unsupported", "released"), times):
            while env.data.time + 1e-8 < target:
                env.step(np.zeros(6))
            renderer.update_scene(env.data, camera=camera, scene_option=option)
            imageio.imwrite(output / f"{name}.png", renderer.render())
