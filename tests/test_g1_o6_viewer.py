"""Space pauses display/control progression; closing produces a logged abort."""

from types import SimpleNamespace

from teleopit.sim.g1_o6 import viewer
from teleopit.sim.g1_o6.config import Config


def test_space_pause_and_close_use_the_logged_runner(monkeypatch, tmp_path):
    import mujoco
    import numpy as np

    callbacks = []
    windows = []
    sleeps = []
    aborts = []

    class Window:
        def __init__(self, key):
            self.key = key
            self.cam = mujoco.MjvCamera()
            self.opt = mujoco.MjvOption()
            self.syncs = 0
            self.running = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.running = False

        def is_running(self):
            return self.running

        def sync(self):
            self.syncs += 1
            if self.syncs in (1, 3):
                self.key(32)
            if self.syncs == 4:
                self.running = False

    def launch(model, data, key_callback):
        window = Window(key_callback)
        windows.append(window)
        return window

    def run(config, output, callback):
        assert output == tmp_path
        env = SimpleNamespace(model=None, data=None, abort=aborts.append)
        callback(env, {})
        callbacks.append(windows[0].syncs)
        callback(env, {})
        return {"final_phase": "ABORT"}

    monkeypatch.setattr(viewer.mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(viewer, "run_trial", run)
    monkeypatch.setattr(viewer.time, "sleep", sleeps.append)
    result = viewer.show(Config(), tmp_path)
    assert callbacks == [3]  # first callback waits through pause/resume
    assert len(sleeps) >= 2
    assert aborts == [["viewer_closed"]]
    assert result["final_phase"] == "ABORT"
    assert not windows[0].opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]
    assert not windows[0].opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT]
    np.testing.assert_allclose(windows[0].cam.lookat, [-0.6, 0, 1.4])
