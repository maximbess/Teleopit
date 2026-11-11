"""Whole-body posture must clear the ladder without extreme wrist flexion."""

import mujoco
import numpy as np
import pytest

from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.o6_diagnostics.posture import install_candidate


@pytest.fixture(scope="module")
def old_candidate():
    env = Environment()
    install_candidate(env)
    return env


def test_wrist_audit_identifies_old_pose_bend(old_candidate):
    from teleopit.sim.g1_o6.posture import posture_metrics

    metrics = posture_metrics(old_candidate)
    assert max(metrics["wrist_bend_deg"].values()) > 70
    assert any(value < 0.02 for value in metrics["wrist_joint_margin_rad"].values())


def test_housing_clearance_uses_all_rungs(old_candidate):
    from teleopit.sim.g1_o6.posture import housing_clearance

    distances = housing_clearance(old_candidate.model, old_candidate.data)
    assert set(distances) == {"left", "right"}
    assert all("rung" in row["obstacle"] for row in distances.values())
    assert all(np.isfinite(row["distance_m"]) for row in distances.values())


def test_balanced_pose_clears_housings_and_preserves_physics():
    from teleopit.sim.g1_o6.grasp_pose import install_balanced
    from teleopit.sim.g1_o6.posture import posture_metrics

    env = Environment()
    arrays = (
        "body_mass",
        "body_inertia",
        "jnt_range",
        "dof_damping",
        "dof_armature",
        "geom_contype",
        "geom_conaffinity",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_ctrlrange",
        "eq_data",
    )
    original = {key: getattr(env.model, key).copy() for key in arrays}
    config = vars(env.config).copy()
    install_balanced(env)
    assert env.initial_errors == []
    assert (
        env.support_targets["left_hand"]
        == env.support_targets["right_hand"]
        == "left_ladder_rung_11"
    )
    metrics = posture_metrics(env)
    assert max(metrics["wrist_bend_deg"].values()) < 30
    assert min(metrics["wrist_joint_margin_rad"].values()) > 1.0
    assert min(r["distance_m"] for r in metrics["housing_clearance"].values()) > 0.03
    for key, value in original.items():
        np.testing.assert_array_equal(getattr(env.model, key), value)
    assert vars(env.config) == config


@pytest.mark.parametrize("reported", [0.0, -0.003])
def test_distance_audit_certifies_separation_without_hiding_penetration(
    monkeypatch, reported
):
    from teleopit.sim.g1_o6.posture import geom_clearance

    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom name="a" type="box" size=".05 .02 .03" euler="0 0 45"/>
    <geom name="b" type="box" size=".05 .02 .03" pos=".085 .085 0" euler="0 0 45"/>
    </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: reported)
    distance, method = geom_clearance(m, d, 0, 1)
    if reported < 0:
        assert distance == reported and method == "native"
    else:
        assert distance == pytest.approx(np.sqrt(2) * 0.085 - 0.1)
        assert method == "separating_plane_lower_bound"


@pytest.mark.parametrize(
    "change",
    [
        {"hand_rungs": (0, 11)},
        {"hand_rungs": (11,)},
        {"grip_offset_m": (0, 0)},
        {"closed_pose": (0,) * 5},
        {"hand_tilt_deg": float("nan")},
        {"hand_spacing_m": 0},
    ],
)
def test_invalid_pose_parameters_are_rejected(change):
    from teleopit.sim.g1_o6.grasp_pose import GraspPose

    with pytest.raises(ValueError):
        GraspPose(**change).validate()


def test_balanced_reset_is_repeatable_and_metrics_are_read_only():
    from teleopit.sim.g1_o6.grasp_pose import GraspPose
    from teleopit.sim.g1_o6.posture import posture_metrics

    env = Environment(pose=GraspPose())
    initial = env.data.qpos.copy()
    controls = env.data.ctrl.copy()
    applied = env.data.qfrc_applied.copy()
    posture_metrics(env)
    np.testing.assert_array_equal(env.data.qpos, initial)
    np.testing.assert_array_equal(env.data.ctrl, controls)
    np.testing.assert_array_equal(env.data.qfrc_applied, applied)
    env.step()
    env.reset()
    np.testing.assert_array_equal(env.data.qpos, initial)
    np.testing.assert_array_equal(env.data.ctrl, controls)
    assert env.data.time == 0 and env.phase == "INITIALIZE"
