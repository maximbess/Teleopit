"""Exports retain the simulated model even after an unrelated compilation."""

import mujoco
import numpy as np
import pytest


@pytest.mark.parametrize("rejected", [False, True])
def test_scene_export_uses_its_environment(tmp_path, rejected):
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6_support.long_hold import prepare_hold
    from teleopit.sim.g1_o6_support.pose_trial import reject_reset

    env = Environment()
    mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
    )
    output = tmp_path / "run"
    if rejected:
        reject_reset(env, output, {}, ["deliberate software-test rejection"])
    else:
        prepare_hold(
            output, env.config.hold_seconds, env.config.physics_dt, None, env, None
        )
    exported = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    assert exported.nq == env.model.nq
    assert exported.ngeom == env.model.ngeom
    for name in dir(env.model):
        value = getattr(env.model, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(exported, name), value, err_msg=name)
    assert exported.opt.timestep == env.model.opt.timestep
