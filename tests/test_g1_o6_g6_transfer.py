"""Upward transfer is a distinct gated protocol with measured recovery."""


def test_adjacent_orders_reach_both_hands_and_keep_thirty_second_recovery():
    from teleopit.sim.g1_o6_support.protocol import transfer_steps

    for side, other in (("left", "right"), ("right", "left")):
        steps = transfer_steps("adjacent", side, 1, False, 10, 10)
        assert [s.side for s in steps if s.name == "REACH"] == [side, other]
        assert "DESTINATION_CAPACITY" in [s.name for s in steps]
        assert steps[-1].name == "POST_HOLD" and steps[-1].minimum_s == 30
        assert not any(s.name in ("RETURN", "APPROACH") for s in steps)
        first = transfer_steps("adjacent-first", side, 1, False, 10, 10)
        assert [s.side for s in first if s.name == "REACH"] == [side]


def test_adjacent_cannot_start_from_same_tread_or_load_only_evidence():
    from teleopit.sim.g1_o6_support.qualification import transition_prerequisites

    request = dict(
        mode="adjacent",
        side="left",
        target_share=0.3,
        cycles=1,
        alternating=False,
        three_seconds=10,
        plateau_seconds=10,
        physics_dt=0.001,
    )
    qualified = [{"metadata": {"experiment": dict(request, mode="load")}}]
    reasons = transition_prerequisites(request, qualified)
    assert "both_orders_same_tread_cycles" in reasons
    assert "first_destination_capacity_qualified" in reasons


def test_g6_loaded_gate_requires_complete_band_convergence_repeat_and_fine():
    from teleopit.sim.g1_o6_support.g6_gates import loaded_hold_gate

    runs = [
        {
            "path": str(i),
            "metadata": {
                "config": {"physics_dt": dt},
                "experiment": {
                    "mode": "load",
                    "target_share": 0.3,
                    "plateau_seconds": 60,
                },
            },
            "summary": {
                "physical_status": "PASS",
                "capacity_status": "PASS",
                "plateau_convergence": {"status": "PASS"},
                "load_measurements": {"continuous_in_band_s": 60},
            },
        }
        for i, dt in enumerate((0.001, 0.001, 0.0005))
    ]
    assert loaded_hold_gate(runs)
    runs[-1]["summary"]["load_measurements"]["continuous_in_band_s"] = 59
    assert not loaded_hold_gate(runs)


def test_capacity_for_old_tread_does_not_cover_a_destination():
    from teleopit.sim.g1_o6_support.qualification import target_capacity_wrenches

    source = {
        "grasp_pose": {},
        "controller": {},
        "support_targets": {"left_hand": "rung7", "right_hand": "rung7"},
        "assets": {},
        "config": {},
    }
    meta = {
        "source_grasp_pose": {},
        "controller": {},
        "source_support_targets": source["support_targets"],
        "assets": {},
        "source_config": {},
        "fixed_base": False,
        "gravity_world_m_s2": [0, 0, -9.81],
        "side": "left",
        "source_sample_support_targets": {"left_hand": "rung8", "right_hand": "rung7"},
    }
    result = {
        "strict_pass": True,
        "plateau_complete": True,
        "physical_retention": True,
        "requested_wrench_world_at_mount": [0, 0, -70, 0, 0, 0],
    }
    tested = target_capacity_wrenches([{"metadata": meta, "summary": result}], source)
    assert "rung7" not in tested["left"]
    assert tested["left"]["rung8"] == [result["requested_wrench_world_at_mount"]]
