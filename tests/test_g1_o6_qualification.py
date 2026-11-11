from copy import deepcopy


def test_identity_invalidates_capacity_after_grasp_or_physics_change():
    from teleopit.sim.g1_o6_support.qualification import identity

    meta = {
        "grasp_pose": {"closed_pose": [0.05, 0.5]},
        "controller": {"preload": 0.12},
        "support_targets": {"left_hand": "rung11"},
        "config": {"hold_seconds": 30, "physics_dt": 0.001, "friction_scale": 1},
        "assets": {"sha256": {"a": "abcd"}},
    }
    changed = deepcopy(meta)
    changed["config"]["hold_seconds"] = 120
    changed["config"]["physics_dt"] = 0.0005
    assert identity(changed) == identity(meta)
    changed["grasp_pose"]["closed_pose"][1] = 0.3
    assert identity(changed) != identity(meta)
    changed = deepcopy(meta)
    changed["config"]["friction_scale"] = 1.5
    assert identity(changed) != identity(meta)


def test_long_gate_requires_nominal_repeat_and_fine_distinct_runs():
    from teleopit.sim.g1_o6_support.qualification import long_hold_gate

    runs = [
        {
            "path": str(i),
            "metadata": {
                "config": {"physics_dt": dt},
                "experiment": {"requested_hold_s": 120},
            },
            "summary": {
                "strict_status": "PASS",
                "drift_assessment": {"status": "PASS"},
                "physics_audit": {
                    "no_external_support": True,
                    "unintended_support_steps": 0,
                    "housing_contact_steps": 0,
                },
            },
        }
        for i, dt in enumerate((0.001, 0.001, 0.0005))
    ]
    assert long_hold_gate(runs)
    assert not long_hold_gate(runs[:2])
    assert not long_hold_gate([runs[0], runs[0], runs[2]])
    runs[2]["summary"]["strict_status"] = "FAIL"
    assert not long_hold_gate(runs)


def test_missing_prerequisites_block_before_creating_a_simulator(tmp_path, monkeypatch):
    import json

    from teleopit.sim.g1_o6_support import transfer

    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "source_run": "absent",
                "long_holds": [],
                "capacity_runs": [],
                "transfers": [],
            }
        )
    )
    monkeypatch.setattr(
        transfer,
        "Environment",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("must not construct a blocked simulation")
        ),
    )
    result = transfer.run_transfer(evidence, tmp_path / "out", mode="release")
    assert result["physical_status"] == "BLOCKED"
    assert not result["progression_justified"]
