"""Bounded G6 candidates preserve physics and expose acquisition clearance."""

from dataclasses import replace

import numpy as np
import pytest


def test_g6_batches_are_bounded_one_family_variants_and_include_lower_treads():
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate, batch_candidates

    base = G6Candidate("control")
    a = batch_candidates("A", base)
    b = batch_candidates("B", a[0])
    c = batch_candidates("C", b[0])
    d = batch_candidates("D", c[0])
    assert len(a) + len(b) + len(c) + len(d) + 1 <= 40
    assert all(x.pose.hand_rungs == (8, 8) for x in a)
    assert any(max(x.pose.hand_rungs) <= 7 for x in b)
    assert all(x.parent == base.id for x in a)
    assert all(x.controller == "reference" for x in a + b + c + d)
    for candidate in a + b + c + d:
        assert G6Candidate.from_dict(candidate.record()).digest() == candidate.digest()
    with pytest.raises(ValueError):
        replace(base, reset={"friction_scale": 0.4})
    with pytest.raises(ValueError):
        replace(base, reset={"root_offset_x": 0.4})


def test_g6_reset_offsets_move_declared_feet_without_changing_compiled_physics():
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    base = make_environment(G6Candidate("base"), 30, 0.001)
    moved = make_environment(
        G6Candidate("feet", reset={"foot_offset_x": -0.01}), 30, 0.001
    )
    for field in (
        "body_mass",
        "body_inertia",
        "geom_size",
        "geom_friction",
        "jnt_range",
        "actuator_forcerange",
    ):
        np.testing.assert_array_equal(
            getattr(moved.model, field), getattr(base.model, field)
        )
    for side in ("left", "right"):
        delta = (
            moved.data.body(side + "_ankle_roll_link").xpos
            - base.data.body(side + "_ankle_roll_link").xpos
        )
        np.testing.assert_allclose(delta, [-0.01, 0, 0], atol=1e-5)
    assert moved.support_targets == base.support_targets


def test_body_clearance_covers_each_knee_shin_housing_forearm_and_all_obstacles():
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_observer import BodyClearance
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    env = make_environment(G6Candidate("control"), 30, 0.001)
    before = env.data.qpos.copy()
    report = BodyClearance(env).sample(env)
    for side in ("left", "right"):
        for part in ("knee", "shin", "housing", "forearm"):
            assert len(report[side + "_" + part]) == 20
    assert len(report["torso"]) == 20
    np.testing.assert_array_equal(env.data.qpos, before)
    # Signed distances are measurements, never collision-mask edits.
    assert env.model.geom("right_ladder_rung_08").contype[0] == 1


def test_missing_anatomical_evidence_cannot_qualify_g6():
    from teleopit.sim.g1_o6_support.g6_gates import initial_pose_gate

    rows = []
    for index, (seconds, dt) in enumerate(
        ((30, 0.001), (60, 0.001), (120, 0.001), (120, 0.001), (120, 0.0005))
    ):
        rows.append(
            {
                "path": str(index),
                "metadata": {
                    "config": {"physics_dt": dt},
                    "experiment": {"requested_hold_s": seconds},
                },
                "summary": {
                    "strict_status": "PASS",
                    "strict": {"unsupported_hold_seconds": seconds},
                    "contact_quality": {"status": "PASS"},
                    "drift_assessment": {"status": "PASS"},
                    "physics_audit": {
                        "no_external_support": True,
                        "unintended_support_steps": 0,
                        "housing_contact_steps": 0,
                    },
                },
            }
        )
    assert initial_pose_gate(rows)
    assert not initial_pose_gate(rows[1:])
    del rows[-1]["summary"]["contact_quality"]
    assert not initial_pose_gate(rows)


def test_phase_contract_exempts_only_the_intentionally_released_hand():
    from teleopit.sim.g1_o6_support.g6_gates import required_contact_sides

    assert required_contact_sides("HOLD", "left") == {"left", "right"}
    assert required_contact_sides("UNLOADED_HOLD", "left") == {"left", "right"}
    assert required_contact_sides("REACH", "left") == {"right"}
    assert required_contact_sides("RELOAD", "left") == {"left", "right"}


def test_adjacent_destination_rejects_top_and_planning_does_not_teleport():
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_reach import (
        adjacent_targets,
        plan_reach,
        next_tread_reach,
    )
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    base = G6Candidate("control")
    env = make_environment(base, 30, 0.001)
    with pytest.raises(ValueError, match="higher tread"):
        adjacent_targets(env)
    lower = replace(base, pose=replace(base.pose, hand_rungs=(7, 7), foot_rungs=(2, 2)))
    env = make_environment(lower, 30, 0.001)
    targets = adjacent_targets(env)
    assert targets["left"] == targets["right"] == "left_ladder_rung_08"
    reachable = next_tread_reach(env)
    assert reachable["status"] in ("REACHABLE", "UNREACHABLE")
    assert reachable["targets"] == targets
    before = [v.copy() for v in (env.data.qpos, env.data.qvel, env.data.ctrl)]
    plan = plan_reach(env, "left", targets["left"], samples=5)
    assert plan["destination"] == targets["left"]
    assert plan["status"] in ("FEASIBLE", "INFEASIBLE")
    assert len(plan["samples"]) == 5
    for actual, expected in zip((env.data.qpos, env.data.qvel, env.data.ctrl), before):
        np.testing.assert_array_equal(actual, expected)
