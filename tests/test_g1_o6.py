import mujoco
import numpy as np
import pytest

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.model import build_model


def test_physical_composition():
    model, _, hands, audit = build_model(Config())
    assert model.nu == 41 and model.nv == 57
    assert sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 1
    assert model.neq == 10
    assert all(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    assert audit["mass_after_kg"] > audit["mass_before_kg"]
    assert 0.6 < audit["mass_after_kg"] - audit["mass_before_kg"] < 0.9
    assert audit["mass_after_kg"] < 40  # fixed ladder mass is not robot weight
    assert len(audit["com_before_world_m"]) == len(audit["com_after_world_m"]) == 3
    assert len(hands) == 2 and model.nsensor >= 4
    assert sum(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) == 1


def test_sensor_sign_and_origin():
    # Parent-on-child wrench about a rotated site, including an offset load.
    xml = """<mujoco><worldbody><body name="child"><inertial mass="1" pos=".1 0 0" diaginertia=".01 .01 .01"/><site name="mount" quat=".707106781 0 .707106781 0"/></body></worldbody><sensor><force name="f" site="mount"/><torque name="t" site="mount"/></sensor></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.xfrc_applied[m.body("child").id, :3] = [0, 0, -2]
    mujoco.mj_forward(m, d)
    rotation = d.site("mount").xmat.reshape(3, 3)
    np.testing.assert_allclose(rotation @ d.sensor("f").data, [0, 0, 11.81], atol=1e-7)
    np.testing.assert_allclose(rotation @ d.sensor("t").data, [0, -1.181, 0], atol=1e-7)


def test_saved_compiled_contact_coefficients(tmp_path):
    model, _, _, _ = build_model(Config(friction_scale=0.25 / 1.8))
    path = tmp_path / "scene.xml"
    mujoco.mj_saveLastXML(str(path), model)
    restored = mujoco.MjModel.from_xml_path(str(path))
    colliding = (restored.geom_contype != 0) | (restored.geom_conaffinity != 0)
    assert np.max(restored.geom_friction[colliding, 0]) <= 0.250001
    assert np.all(restored.geom_condim[colliding] == 4)
    visual = np.array(
        [restored.geom(i).name.endswith("_visual") for i in range(restored.ngeom)]
    )
    assert not np.any(colliding[visual])


def test_export_preserves_signed_wrench_and_mount_reference(tmp_path):
    import json

    from teleopit.sim.g1_o6.run import export_traces
    from teleopit.sim.o6_grasp.replay import WristLoadReplay

    wrench = [1.0, 2.0, -30.0, -0.3, 0.2, 0.1]
    rows = [
        {"time": t, "wrist_world_on_hand": {s: wrench for s in ("left", "right")}}
        for t in (0, 0.02)
    ]
    export_traces(
        rows,
        {"code_revision": "test", "config": {}},
        {"final_phase": "ABORT"},
        tmp_path,
    )
    for side in ("left", "right"):
        path = tmp_path / (side + ".wrist.json")
        trace = WristLoadReplay.load(path)
        np.testing.assert_array_equal(trace.sample(0.01), wrench)
        assert trace.frame == "world" and trace.application_site == "wrist_mount"
        np.testing.assert_array_equal(trace.point_m, [0, 0, 0])
        assert json.loads(path.read_text())["source"]["kind"] == "simulation"


@pytest.fixture(scope="module")
def coupled():
    from teleopit.sim.g1_o6.env import Environment

    return Environment()


def test_reachable_reset_and_independent_bounded_control(coupled):
    env = coupled
    row = env.reset()
    assert row["penetration_m"] == 0
    assert all(v["position_error_m"] < 0.005 for v in env.ik.values())
    assert all(v["angle_error_rad"] < 0.1 for v in env.ik.values())
    assert row["joint_limit_errors"] == {}
    body_indices = np.arange(29)
    body = env.data.ctrl[body_indices].copy()
    env.phase = "ACQUIRE"
    env.data.time = 1.0
    env.commands()
    np.testing.assert_allclose(env.data.ctrl[body_indices], body)
    assert np.any(env.data.ctrl[env.hand_act["left"]] != 0)
    assert np.any(env.data.ctrl[env.hand_act["right"]] != 0)
    limits = env.model.jnt_range[env.model.actuator_trnid[:, 0]]
    assert np.all(env.data.ctrl >= limits[:, 0])
    assert np.all(env.data.ctrl <= limits[:, 1])


def test_invalid_initial_state_aborts_in_environment(coupled):
    env = coupled
    env.reset()
    env.initial_errors = ["unreachable_grasp"]
    row = env.step()
    assert env.phase == "ABORT"
    assert env.abort_reasons == ["unreachable_grasp"]
    assert row["next_phase"] == "ABORT"


def test_retraction_clearance_uses_physical_distance(coupled):
    env = coupled
    env.reset()
    assert not env.clearance()
    # Test-only placement, never used by the experiment's controller.
    env.data.qpos[0] -= 0.3
    mujoco.mj_forward(env.model, env.data)
    assert env.clearance()


def test_both_release_sides_shift_then_return_only_selected_arm(coupled):
    from dataclasses import replace

    env = coupled
    env.reset()
    for side, sign in (("left", 1), ("right", -1)):
        env.config = replace(Config(), release_side=side)
        waist = env.model.actuator("waist_roll_joint").id
        shoulder = env.model.actuator(side + "_shoulder_pitch_joint").id
        other = env.model.actuator(
            ("right" if side == "left" else "left") + "_shoulder_pitch_joint"
        ).id
        env.phase = "UNLOAD_SELECTED_HAND"
        target = env.body_target.copy()
        env.transfer_targets(target, 3)
        assert target[waist] - env.body_target[waist] == pytest.approx(sign * 0.04)
        env.phase = "RETRACT"
        target = env.body_target.copy()
        env.transfer_targets(target, 3)
        assert target[shoulder] - env.body_target[shoulder] == pytest.approx(0.15)
        assert target[other] == env.body_target[other]
        env.phase = "RETURN"
        target = env.body_target.copy()
        env.transfer_targets(target, 3)
        assert target[shoulder] == env.body_target[shoulder]
    env.config = Config()


def test_support_requires_correct_partner_and_opposing_contacts():
    from teleopit.sim.g1_o6.diagnostics import support, unintended

    def contact(partner, normal):
        return {
            "regions": ["environment", "left_hand"],
            "geoms": [partner, "left_o6_finger"],
            "normal_force": 3,
            "force_on_second_world": normal,
            "normal_on_second_world": normal,
            "slip_m_s": 0.0,
        }

    rung = "left_ladder_rung_08"
    wrong = [contact("left_ladder_rail", [0, 0, 1])]
    assert not support(wrong, "left_hand", rung)["loaded"]
    assert unintended(wrong)
    right = [contact(rung, [0, 0, 1]), contact(rung, [0, 0, -1])]
    assert support(right, "left_hand", rung)["opposing"]
    assert not unintended(right)
    assert not support(right[:1], "left_hand", rung)["opposing"]
