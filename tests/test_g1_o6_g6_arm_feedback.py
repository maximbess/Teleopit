"""Behavioral bounds and immutable identity for the lowered G6 experiment."""

import json
from dataclasses import replace

import numpy as np
import pytest


def test_four_feedback_variants_are_lowered_bounded_and_round_trip():
    from teleopit.sim.g1_o6_support.g6_arm import ArmFeedback
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import candidates

    rows = candidates()
    assert len(rows) == 5 and rows[0].controller == "reference"
    assert {(r.controller.reference, r.controller.rate_rad_s) for r in rows[1:]} == {
        (mode, rate) for mode in ("retain", "seek") for rate in (.005, .02)}
    for candidate in rows:
        assert candidate.pose.hand_rungs == (7, 7)
        assert candidate.pose.foot_rungs == (2, 2)
        assert candidate.reset == {"root_offset_x": -.04}
        restored = G6Candidate.from_dict(json.loads(json.dumps(candidate.record())))
        assert restored.digest() == candidate.digest()
    for changes in ({"rate_rad_s": .021}, {"rate_rad_s": float("nan")},
                    {"reference": "unbounded"}, {"rate_rad_s": 0}):
        with pytest.raises(ValueError):
            replace(ArmFeedback(), **changes)


def test_feedback_starts_at_removal_and_changes_only_bounded_arm_targets():
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    env = make_environment(candidates()[2], 5, .001)
    controller = env.hold_controller
    env.phase, env.data.time = "ACQUIRE", 2.0
    np.testing.assert_array_equal(controller.desired(env, 1), controller.base.desired(env, 1))
    assert not controller.reference
    env.phase, env.data.time = "REMOVE_ASSISTANCE", 3.04
    first = controller.desired(env, 1)
    np.testing.assert_allclose(first, controller.base.desired(env, 1), atol=1e-12)
    assert controller.latch_time == 3.04
    # Synthetic displacement tests the law, not physical qualification.
    env.data.qpos[0] += .01
    fields = ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied", "qacc",
              "qacc_warmstart", "site_xpos", "actuator_force")
    before = {name: getattr(env.data, name).copy() for name in fields}
    next_target = controller.desired(env, 1)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(env.data, name), value)
    change = abs(next_target - first)
    indices = np.concatenate(list(controller.indices.values()))
    assert 0 < change.max() <= .02 * .02 + 1e-12
    assert np.all(change[np.setdiff1d(np.arange(env.model.nu), indices)] == 0)
    assert max(controller.state()["command_displacement_m"].values()) <= .005 + 1e-9
    env.reset()
    assert not env.hold_controller.reference
    assert env.hold_controller.latch_time is None


def test_cartesian_bound_checks_actual_scratch_kinematics_and_preserves_effort_freeze():
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.pose_arm import bounded_correction

    env = make_environment(candidates()[2], 5, .001)
    controller = env.hold_controller
    side = "left"
    indices = controller.indices[side]
    desired = controller.base.desired(env, 1)
    offset = np.full(len(indices), .03)
    limited, distance, angle = controller.project_command(env, desired, side, offset)
    assert np.linalg.norm(limited) < np.linalg.norm(offset)
    assert distance <= .005 + 1e-9 and angle <= .03 + 1e-9
    frozen = bounded_correction(np.zeros(2), np.ones(2), np.zeros(2),
        np.array([[-1, 1], [-1, 1]]), np.array([0, .98]), .02, rate_limit=.02)
    np.testing.assert_allclose(frozen, [.0004, 0])


def test_acquisition_seek_uses_intended_thumb_gap_with_a_fixed_five_mm_budget():
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    env = make_environment(candidates()[4], 5, .001)
    env.phase, env.phase_start, env.data.time = "ACQUIRE", .02, 2.02
    controller = env.hold_controller
    controller.desired(env, 1)
    state = controller.state()
    assert controller.reference and controller.latch_time == 2.02
    assert set(state["thumb_seek"]) == {"left", "right"}
    for sample in state["thumb_seek"].values():
        assert sample["target"] == "left_ladder_rung_07"
        assert np.linalg.norm(sample["target_shift_rung_m"]) <= .005 + 1e-12
    anchor = {s: p.copy() for s, (p, _) in controller.anchor.items()}
    env.data.qpos[0] += .002
    controller.desired(env, 1)
    for side, position in anchor.items():
        np.testing.assert_array_equal(controller.anchor[side][0], position)


def test_feedback_scene_reconstruction_keeps_wrapper_and_original_physics():
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.long_hold import experiment_metadata
    from teleopit.sim.g1_o6_support.scene_contract import canonical, restore_environment

    env = make_environment(candidates()[1], 5, .001)
    metadata = canonical(experiment_metadata(env, snapshot(env), {"mode": "long-hold"}))
    assert metadata["controller"]["schema"] == "g6_acquisition_arm_feedback_v1"
    assert metadata["environment_spec"]["arm_feedback"]["reference"] == "retain"
    restored = restore_environment(metadata, .0005)
    assert restored.hold_controller.settings == env.hold_controller.settings
    assert restored.config.control_dt == .02 and restored.model.opt.timestep == .0005
    for field in ("geom_friction", "geom_size", "body_mass", "body_inertia", "jnt_range", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(restored.model, field), getattr(env.model, field))


def test_repeated_feedback_and_backoff_preserve_rate_and_total_command_bounds():
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    env = make_environment(candidates()[2], 5, .001)
    controller = env.hold_controller
    env.phase, env.data.time = "REMOVE_ASSISTANCE", 3.04
    previous = controller.desired(env, 1)
    origin = env.data.qpos[:3].copy()
    arms = np.concatenate(list(controller.indices.values()))
    for index in range(100):
        env.data.time += .02
        env.data.qpos[:3] = origin + [.012 * np.cos(index / 20), 0, .01 * np.sin(index / 20)]
        command = controller.desired(env, 1)
        assert np.max(abs(command[arms] - previous[arms])) <= .0004 + 1e-10
        assert max(controller.displacement.values()) <= .005 + 1e-9
        assert max(controller.angle.values()) <= .03 + 1e-9
        assert max(np.max(abs(offset)) for offset in controller.offset.values()) <= .03
        previous = command
    prior_norm = sum(np.linalg.norm(offset) for offset in controller.offset.values())
    env.data.qpos[:3] = origin + [.04, 0, 0]
    command = controller.desired(env, 1)
    assert "drift" in controller.inhibition
    assert sum(np.linalg.norm(offset) for offset in controller.offset.values()) < prior_norm
    assert np.max(abs(command[arms] - previous[arms])) <= .0004 + 1e-10


def test_unreliable_thumb_geometry_never_becomes_an_invented_contact_target(monkeypatch):
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support import g6_arm

    env = make_environment(candidates()[3], 5, .001)
    monkeypatch.setattr(g6_arm.mujoco, "mj_geomDistance", lambda *args: 0)
    env.phase, env.phase_start, env.data.time = "ACQUIRE", .02, 2.02
    env.hold_controller.desired(env, 1)
    for record in env.hold_controller.seek.values():
        assert record["status"] == "NO_RELIABLE_GAP"
        assert record["target_shift_rung_m"] == [0, 0, 0]


def test_loaded_thumb_is_not_retargeted_to_repair_a_missing_finger(monkeypatch):
    from teleopit.sim.g1_o6_support import g6_arm
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    env = make_environment(candidates()[3], 5, .001)
    env.phase, env.phase_start, env.data.time = "ACQUIRE", .02, 2.02
    controller = env.hold_controller
    controller.desired(env, 1)
    controller.quality["left"].update(thumb_normal_N=1.0, valid=False)
    monkeypatch.setattr(g6_arm, "thumb_gap", lambda *args: {
        "target": "left_ladder_rung_07", "status": "MEASURED", "vector_world_m": [0, 0, .003]})
    controller.seek_reference(env, "left")
    assert controller.seek["left"]["target_shift_rung_m"] == [0, 0, 0]


def test_command_projection_measures_physically_clipped_servo_targets():
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    env = make_environment(candidates()[2], 5, .001)
    controller = env.hold_controller
    indices = controller.indices["left"]
    desired = controller.base.desired(env, 1)
    joint = env.model.actuator_trnid[indices[0], 0]
    desired[indices[0]] = env.model.jnt_range[joint, 1] + .04
    outward = np.zeros(len(indices))
    outward[0] = .03
    _, distance, angle = controller.project_command(env, desired, "left", outward)
    # Both commands clip to the same physical joint limit.
    assert distance < 1e-12 and angle < 1e-12


@pytest.mark.parametrize("changes", [
    {"id": "invalid/name"}, {"controller": "unbounded"},
    {"reset": {"root_offset_x": float("nan")}},
    {"reset": {"root_offset_x": .081}}, {"reset": {"unknown": 0}},
    {"pose": {"hand_rungs": [9, 9], "foot_rungs": [2, 2]}},
])
def test_feedback_candidate_rejects_out_of_domain_inputs(changes):
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import candidates

    with pytest.raises(ValueError):
        G6Candidate.from_dict({**candidates()[1].record(), **changes})
