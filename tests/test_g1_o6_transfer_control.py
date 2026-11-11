import numpy as np


def test_bounded_feedback_requires_measured_error_and_stops_at_effort_limit():
    from teleopit.sim.g1_o6_support.control import bounded_feedback

    jac = np.eye(2)
    zero = np.zeros(2)
    update = bounded_feedback(zero, jac, np.array([2, -1]), np.ones(2), zero, 0.02)
    np.testing.assert_allclose(update, [-0.0001, 0.0001])
    np.testing.assert_array_equal(
        bounded_feedback(zero, jac, zero, np.ones(2), zero, 0.02), zero
    )
    np.testing.assert_array_equal(
        bounded_feedback(zero, jac, np.ones(2), np.ones(2), np.ones(2), 0.02), zero
    )
    offset = np.full(2, 0.08)
    assert (
        np.max(bounded_feedback(offset, jac, -np.ones(2), np.ones(2), zero, 0.02))
        <= 0.08
    )


def test_kinematic_planning_and_clearance_leave_live_state_unchanged():
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6_support.kinematics import Clearance, arm_target

    env = Environment()
    before = env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy()
    site = env.data.site(env.hands["left"]["mount"])
    result = arm_target(env, "left", site.xpos.copy(), site.xmat.reshape(3, 3).copy())
    assert result["position_error_m"] < 1e-6
    geometry = Clearance(env, "left")
    assert env.support_targets["left_hand"] == "left_ladder_rung_11"
    assert any(
        env.model.geom(g).name == "left_ladder_rung_11" for _, g in geometry.pairs
    )
    assert geometry.measure(env.data)["clearance_m"] <= 0.1
    for actual, saved in zip((env.data.qpos, env.data.qvel, env.data.ctrl), before):
        np.testing.assert_array_equal(actual, saved)


def test_selected_opening_leaves_other_fingers_under_the_qualified_controller():
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6_support.control import TransferController
    from teleopit.sim.g1_o6_support.protocol import Protocol, Step

    env = Environment()
    original = env.hold_controller
    protocol = Protocol([Step("OPEN", 3, 15, "opened", "right")])
    controller = TransferController(env, original, protocol, 0.05)
    env.phase = "HOLD"  # Controller unit test: no physical stepping or qualification.
    controller.latest = controller.baseline = snapshot(env)
    controller.enter(env, controller.latest)
    controller.open_start["right"] = np.full(6, 0.5)
    protocol.started = -3
    expected = original.desired(env, 1)
    actual = controller.desired(env, 1)
    np.testing.assert_array_equal(
        actual[env.hand_act["left"]], expected[env.hand_act["left"]]
    )
    np.testing.assert_array_equal(actual[env.hand_act["right"]], np.zeros(6))
    assert (
        abs(
            sum(controller.target_forces.values())
            - sum(v["force_world"][2] for v in controller.baseline["supports"].values())
        )
        < 1e-10
    )


def test_read_only_audit_detects_actual_applied_arrays_and_optional_contact_loss():
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6_support.audit import PhysicsAudit

    env = Environment()
    env.phase = "HOLD"
    env.data.xfrc_applied[env.model.body("pelvis").id, 2] = 0.1
    before = env.data.qpos.copy()
    audit = PhysicsAudit(env)
    audit.optional_supports = {"left_hand"}
    audit(env)
    np.testing.assert_array_equal(env.data.qpos, before)
    report = audit.report()
    assert not report["no_external_support"]
    assert "external_applied_force" in report["first_crossings"]
    assert "contact_loss:left_hand" not in report["first_crossings"]
    assert "contact_loss:right_hand" in report["first_crossings"]
