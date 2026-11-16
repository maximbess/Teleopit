"""State-specific capacity, native contact observation and immutable scheduling."""

import numpy as np

from teleopit.sim.g1_o6_support.capacity import sampled_state_key
from teleopit.sim.g1_o6_support.qualification import critical_groups, state_wrenches


def sample():
    return {
        "time": 6.0,
        "phase": "HOLD",
        "assistance_gain": 0,
        "base_pose": [0, 0, 1, 1, 0, 0, 0],
        "joint_positions": [0.0, 0.2],
        "wrist_world_on_hand": {s: [0, 0, -10, 0, 0.1, 0] for s in ("left", "right")},
    }


def test_capacity_requires_the_sampled_state_and_correct_target():
    row = sample()
    targets = {s + "_hand": "left_ladder_rung_07" for s in ("left", "right")}
    cap = {
        "metadata": {
            "side": "left",
            "source_sample_support_targets": targets,
            "source_sample": {"state_sha256": sampled_state_key(row, targets)},
        },
        "summary": {
            "requested_wrench_world_at_mount": [0, 0, -12, 0, 0.12, 0],
            "strict_pass": True,
            "physical_retention": True,
            "plateau_complete": True,
        },
    }
    args = ([row], "left", targets["left_hand"], {"time_s": 6.0}, targets, [cap])
    assert state_wrenches(*args) == [cap["summary"]["requested_wrench_world_at_mount"]]
    row["joint_positions"][1] += 0.01
    assert state_wrenches(*args) == []


def test_critical_demand_groups_ignore_acquisition_and_released_hand():
    row = sample()
    targets = {s + "_hand": "left_ladder_rung_07" for s in ("left", "right")}
    moving = {**row, "experiment_phase": "REACH", "transfer": {"side": "left"}}
    groups = critical_groups([{**row, "phase": "ACQUIRE"}, moving], targets)
    assert set(groups) == {("right", targets["right_hand"], "REACH")}


def test_g6_observer_writes_actual_contacts_and_keeps_physics_state(tmp_path):
    import gzip
    import json
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.g6_observer import G6Observer, native_friction

    env = make_environment(G6Candidate("fixture"), 0.02, 0.001)
    env.step()
    observer = G6Observer(env, tmp_path)
    before = env.data.qpos.copy()
    observer(env)
    np.testing.assert_array_equal(before, env.data.qpos)
    observer.close()
    with gzip.open(tmp_path / "physics_contacts.jsonl.gz", "rt") as stream:
        row = json.loads(stream.readline())
    assert row["support_targets"] == env.support_targets
    assert row["contacts"]
    for contact in row["contacts"]:
        if any("_ladder_rung_" in name for name in contact["geoms"]):
            assert contact["friction"] == [1.8, 1.8, 0.02, 0.002, 0.002]
    observed = {
        s + "_o6_finger|left_ladder_rung_07": {
            "coefficients": [[1.8, 1.8, 0.02, 0.002, 0.002]]
        }
        for s in ("left", "right")
    }
    assert native_friction(observed)["status"] == "PASS"
    observed["left_o6_finger|left_ladder_rung_07"]["coefficients"][0][0] = 0.8
    assert native_friction(observed)["status"] == "FAIL"


def test_fine_step_capacity_matches_same_physical_scene():
    from teleopit.sim.g1_o6_support.qualification import capacity_scene_matches

    source = {"physical_identity": {"physics_dt": 0.001, "body": "hash"}}
    capacity = {"source_physical_identity": {"physics_dt": 0.0005, "body": "hash"}}
    assert capacity_scene_matches(capacity, source)
    capacity["source_physical_identity"]["body"] = "different"
    assert not capacity_scene_matches(capacity, source)


def test_pipeline_schedule_has_ordered_loading_and_all_transfer_confirmations():
    from teleopit.sim.g1_o6_support.g6_stages import transfer_schedule
    from teleopit.sim.g1_o6_support.transfer import validate_request

    schedule = transfer_schedule()
    assert [r["target_share"] for r in schedule[:4]] == [0.05, 0.1, 0.2, 0.3]
    defaults = dict(
        side="left",
        cycles=1,
        alternating=False,
        three_seconds=1,
        plateau_seconds=10,
        physics_dt=0.001,
    )
    for request in schedule:
        validate_request({**defaults, **request})
    for side in ("left", "right"):
        nominal = [
            r
            for r in schedule
            if r["mode"] == "adjacent"
            and r["side"] == side
            and r.get("physics_dt", 0.001) == 0.001
        ]
        fine = [
            r
            for r in schedule
            if r["mode"] == "adjacent"
            and r["side"] == side
            and r.get("physics_dt") == 0.0005
        ]
        assert len(nominal) == 3 and len(fine) == 1
