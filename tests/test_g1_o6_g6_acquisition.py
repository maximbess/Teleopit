"""Named thumb articulation, bounded scheduling and unchanged physical gates."""

from dataclasses import replace
import json

import numpy as np
import pytest


def test_named_thumb_targets_follow_actual_joint_names_and_native_ranges():
    from teleopit.sim.g1_o6.grasp_pose import GraspPose
    from teleopit.sim.o6_hand import ACTIVE_NAMES
    from teleopit.sim.g1_o6_support.g6_acquisition import thumb_pose
    from teleopit.sim.g1_o6_support.g6_acquisition_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    named = dict(zip(ACTIVE_NAMES, thumb_pose(GraspPose(), pitch=.25, yaw=.5).closed_pose))
    assert named["rh_thumb_cmc_pitch"] == .25
    assert named["rh_thumb_cmc_yaw"] == .5
    candidate = candidates()[1]
    env = make_environment(candidate, 5, .001)
    for side, hand in env.hands.items():
        targets = dict(zip(hand["active"], env.pose.closed_pose))
        assert next(v for k, v in targets.items() if k.endswith("thumb_cmc_pitch")) == .25
        assert next(v for k, v in targets.items() if k.endswith("thumb_cmc_yaw")) == .5
        np.testing.assert_array_equal(env.model.jnt_range[env.model.actuator_trnid[env.hand_act[side], 0]],
            [[0, .58], [0, 1.36], [0, 1.6], [0, 1.6], [0, 1.6], [0, 1.6]])
    assert thumb_pose(candidates()[0].pose, pitch=.25, yaw=.5).closed_pose[2:] == candidates()[0].pose.closed_pose[2:]
    for values in ({"pitch": .581, "yaw": .5}, {"pitch": .25, "yaw": 1.361}, {"pitch": float("nan"), "yaw": .5}):
        with pytest.raises(ValueError):
            thumb_pose(env.pose, **values)


def test_thumb_first_schedule_uses_original_preload_and_only_changes_finger_targets():
    from teleopit.sim.g1_o6_support.g6_acquisition_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    env = make_environment(candidates()[3], 5, .001)
    controller = env.hold_controller
    env.phase, env.phase_start, env.data.time = "ACQUIRE", .02, .52
    before = {name: getattr(env.data, name).copy() for name in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")}
    baseline = controller.base.desired(env, .2)
    command = controller.desired(env, .2)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(env.data, name), value)
    fingers = np.concatenate(list(env.hand_act.values()))
    np.testing.assert_array_equal(command[np.setdiff1d(np.arange(env.model.nu), fingers)], baseline[np.setdiff1d(np.arange(env.model.nu), fingers)])
    for indices in env.hand_act.values():
        np.testing.assert_array_equal(command[indices[2:]], 0)
        assert np.all(command[indices[:2]] > 0)
        assert np.max(command[indices] - env.data.qpos[env.act_qpos[indices]]) <= env.pose.finger_preload_Nm / controller.base.kp[indices].min()
    env.data.time = 3.02
    np.testing.assert_allclose(controller.desired(env, 1), controller.base.desired(env, 1))
    env.phase = "HOLD"
    np.testing.assert_array_equal(controller.desired(env, 1), controller.base.desired(env, 1))


def test_acquisition_requires_both_loaded_ordered_opposed_hands_for_half_a_second():
    from teleopit.sim.g1_o6_support.g6_acquisition_audit import AcquisitionAudit

    audit = AcquisitionAudit(.001, .5)
    good = {side: {"quality": {"valid": True, "thumb_normal_N": 1., "minimum_finger_normal_N": .5}} for side in ("left", "right")}
    for index in range(499):
        audit.observe("ACQUIRE", good, [])
    assert not audit.report()["ready_for_removal"]
    audit.observe("ACQUIRE", good, [])
    assert audit.report()["ready_for_removal"]
    bad = {**good, "right": {"quality": {"valid": False, "thumb_normal_N": 0., "minimum_finger_normal_N": .5}}}
    audit.observe("ACQUIRE", bad, [])
    assert not audit.report()["ready_for_removal"]
    for index in range(500):
        audit.observe("REMOVE_ASSISTANCE", good, [])
    assert not audit.report()["ready_for_removal"]
    assert audit.report()["ready_for_hold"]
    fresh = AcquisitionAudit(.001, .5)
    for phase in ("ACQUIRE", "REMOVE_ASSISTANCE"):
        for index in range(500):
            fresh.observe(phase, good, [])
    assert fresh.report()["ready_for_hold"]
    fresh.observe("REMOVE_ASSISTANCE", good, ["penetration"])
    assert not fresh.report()["ready_for_hold"]


def test_acquisition_controller_identity_scene_restore_and_feedback_bounds():
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.g6_acquisition_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_acquisition import GraspAcquisition
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.long_hold import experiment_metadata
    from teleopit.sim.g1_o6_support.scene_contract import canonical, restore_environment

    candidate = replace(candidates()[3], controller=GraspAcquisition("thumb_first", .005))
    restored = G6Candidate.from_dict(json.loads(json.dumps(candidate.record())))
    assert restored.digest() == candidate.digest()
    assert replace(candidate, controller=GraspAcquisition("thumb_first", .02)).digest() != candidate.digest()
    env = make_environment(candidate, 5, .001)
    metadata = canonical(experiment_metadata(env, snapshot(env), {"mode": "long-hold"}))
    fine = restore_environment(metadata, .0005)
    assert fine.grasp_acquisition == env.grasp_acquisition
    assert fine.config.control_dt == .02
    for field in ("geom_friction", "body_mass", "body_inertia", "jnt_range", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(env.model, field), getattr(fine.model, field))
    env.phase, env.data.time = "REMOVE_ASSISTANCE", 3.04
    env.hold_controller.desired(env, 1)
    assert not env.hold_controller.reference
    env.phase, env.data.time = "HOLD", 5.04
    original = env.hold_controller.desired(env, 1)
    env.data.qpos[0] += .01
    adjusted = env.hold_controller.desired(env, 1)
    assert 0 < np.max(abs(adjusted - original)) <= .005 * .02 + 1e-12
    assert "grasp_acquisition" in env.hold_controller.state()
    for args in (("unknown", 0), ("thumb_first", .03), ("thumb_first", float("nan"))):
        with pytest.raises(ValueError):
            GraspAcquisition(*args)
