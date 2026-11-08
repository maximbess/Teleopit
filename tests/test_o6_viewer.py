"""Viewer controls must stop physics while continuing window synchronization."""

from contextlib import nullcontext
from types import SimpleNamespace

import mujoco
import numpy as np

from teleopit.sim.o6_grasp import GraspConfig
from teleopit.sim.o6_grasp import viewer as viewer_module


def test_space_pauses_and_resumes_without_advancing_physics(monkeypatch):
    frames = []
    labels = []
    options = []
    callback = None
    data = None
    scene = SimpleNamespace(ngeom=0, geoms=[mujoco.MjvGeom()])
    window = SimpleNamespace(
        cam=mujoco.MjvCamera(),
        opt=mujoco.MjvOption(),
        user_scn=scene,
        lock=nullcontext,
        is_running=lambda: len(frames) < 5,
    )

    def sync():
        frames.append((data.time, data.qpos.copy(), data.ctrl.copy()))
        labels.append(scene.geoms[0].label)
        options.append(
            (
                window.opt.frame,
                window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE],
                window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT],
            )
        )
        if callback is not None:
            if len(frames) in (1, 3):
                callback(32)
            if len(frames) == 2:
                callback(ord("A"))  # unrelated keys do not resume

    def launch(model, simulation_data, *, key_callback=None):
        nonlocal callback, data
        callback, data = key_callback, simulation_data
        window.sync = sync
        return nullcontext(window)

    monkeypatch.setattr(viewer_module.mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(viewer_module.time, "sleep", lambda _: None)
    viewer_module.show(GraspConfig())
    assert callback is not None
    np.testing.assert_allclose(
        [frame[0] for frame in frames], [0.02, 0.02, 0.02, 0.04, 0.06]
    )
    for frame in frames[1:3]:
        np.testing.assert_array_equal(frame[1], frames[0][1])
        np.testing.assert_array_equal(frame[2], frames[0][2])
    assert "PAUSED" in labels[1] and "PAUSED" in labels[2]
    assert "PAUSED" not in labels[3]
    assert all(
        frame == (mujoco.mjtFrame.mjFRAME_NONE, False, False) for frame in options
    )
