"""Compiled geometry, reset adaptation and unchanged-reference behavior."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest


def compile_ladder(identity):
    from teleopit.sim.ladder_geometry import add_ladder
    from teleopit.sim.ladder_config import geometry_matrix, compiled_ladder

    config = geometry_matrix()[identity]
    spec = mujoco.MjSpec()
    add_ladder(spec, config)
    model = spec.compile()
    return config, model, compiled_ladder(model, config)


def test_dimension_controls_keep_reference_centers_and_rails():
    _, baseline, _ = compile_ladder("G0")
    for identity, dimensions in (("G1", (.110, .700, .0381)),
                                 ("G2", (.080, .700, .070)),
                                 ("G3", (.080, .700, .0381))):
        _, model, report = compile_ladder(identity)
        np.testing.assert_array_equal(model.geom_pos, baseline.geom_pos)
        np.testing.assert_array_equal(model.geom_quat, baseline.geom_quat)
        np.testing.assert_allclose(model.geom("left_ladder_rung_11").size * 2, dimensions)
        np.testing.assert_array_equal(model.geom("left_ladder_rail_01").size,
                                      baseline.geom("left_ladder_rail_01").size)
        assert report["rungs"]["left_ladder_rung_11"]["full_dimensions_xyz_m"] == list(dimensions)


def test_pitch_control_anchors_foot_rung_four():
    _, baseline, _ = compile_ladder("G0")
    _, model, report = compile_ladder("G4")
    np.testing.assert_allclose(model.geom("left_ladder_rung_04").pos,
                               baseline.geom("left_ladder_rung_04").pos, atol=1e-15)
    assert report["front_pitch"]["along_rail_m"] == pytest.approx(.25)
    assert report["front_pitch"]["vertical_m"] == pytest.approx(.25 * 2.8 / np.hypot(2.8, 1.05))


def test_ladderman_and_commercial_are_distinct_resolved_geometries():
    _, model, report = compile_ladder("G5")
    assert len(report["rungs"]) == 6
    np.testing.assert_allclose(model.geom("left_ladder_rung_01").pos, [-.58, 0, .19095])
    assert report["front_pitch"]["along_rail_m"] == pytest.approx(.25)
    _, model, report = compile_ladder("G6")
    assert len(report["rungs"]) == 16
    assert len(report["rails"]) == 4
    assert report["front_pitch"]["vertical_m"] == pytest.approx(.265)
    assert report["front_pitch"]["along_rail_m"] == pytest.approx(.265 / np.sin(np.deg2rad(70)))
    for name, rung in report["rungs"].items():
        assert model.geom(name).contype[0] == 1
        assert rung["position_world_m"][2] - rung["full_dimensions_xyz_m"][2] / 2 > 0
    assert report["overall_bounds_xyz_m"][1][1] - report["overall_bounds_xyz_m"][0][1] == pytest.approx(.59)


def test_invalid_geometry_is_rejected_at_boundary():
    from teleopit.sim.ladder_config import geometry_matrix

    base = geometry_matrix()["G6"]
    for change in ({"pitch_m": 0}, {"pitch_convention": "ambiguous"},
                   {"inclination_deg": float("nan")}, {"rung_count": 0}):
        with pytest.raises(ValueError):
            replace(base, **change)


def test_explicit_g0_preserves_default_compiled_model_reset_and_commands():
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.ladder_config import geometry_matrix

    original = Environment()
    explicit = Environment(ladder=geometry_matrix()["G0"])
    for name in dir(original.model):
        value = getattr(original.model, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(explicit.model, name), err_msg=name)
    np.testing.assert_array_equal(original.data.qpos, explicit.data.qpos)
    for _ in range(3):
        original.step()
        explicit.step()
    np.testing.assert_array_equal(original.data.qpos, explicit.data.qpos)
    np.testing.assert_array_equal(original.data.ctrl, explicit.data.ctrl)


def test_support_indices_validate_against_actual_ladder():
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6.grasp_pose import GraspPose
    from teleopit.sim.ladder_config import geometry_matrix
    from teleopit.sim.g1_o6_support.pose_trial import reset_report

    pose = GraspPose(hand_rungs=(6, 6), foot_rungs=(1, 1))
    env = Environment(pose=pose, ladder=geometry_matrix()["G5"])
    assert env.support_targets["right_foot"] == "left_ladder_rung_01"
    report = reset_report(env)
    assert "11" not in report["rung_transforms"]
    assert report["rung_11_minus_10_world_m"] is None
    with pytest.raises(ValueError, match="support rung"):
        Environment(ladder=geometry_matrix()["G5"])


def test_robot_invariants_allow_ladder_changes_and_detect_robot_changes():
    from teleopit.sim.g1_o6.config import Config
    from teleopit.sim.g1_o6.model import build_model
    from teleopit.sim.g1_o6_support.ladder_identity import physical_identity, assert_preserved
    from teleopit.sim.ladder_config import geometry_matrix

    baseline = physical_identity(build_model(Config())[0])
    for geometry in geometry_matrix().values():
        model = build_model(Config(), ladder=geometry)[0]
        assert_preserved(physical_identity(model), baseline)
    model.body_mass[model.body("pelvis").id] += .001
    with pytest.raises(ValueError, match="undeclared"):
        assert_preserved(physical_identity(model), baseline)
