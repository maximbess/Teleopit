"""A full physical hold through the public runner, including its evidence."""


def test_balanced_preset_completes_strict_hold_without_housing_support(tmp_path):
    from teleopit.sim.g1_o6.config import Config
    from teleopit.sim.g1_o6.run import run_trial

    physics = []

    def observe(env):
        if env.phase == "HOLD":
            physics.append(
                (abs(env.data.qfrc_applied).max(), abs(env.data.xfrc_applied).max())
            )

    result = run_trial(
        Config(), tmp_path, posture="thumb-below-balanced", physics_observer=observe
    )
    assert len(physics) >= 10000
    assert max(max(forces) for forces in physics) == 0
    assert result["static_hold_passed"]
    assert result["unsupported_hold_seconds"] >= 10
    assert result["final_phase"] == "HOLD_COMPLETE"
    assert not result["release_reacquire_passed"]
    posture = result["posture"]
    assert posture["housing_contact_samples"] == 0
    assert posture["hold_thumb_below_fingers_above"]
    assert posture["max_hold_wrist_bend_deg"] < 30
    assert posture["min_hold_housing_clearance_m"] > 0.002


def test_empty_hold_evidence_is_not_a_pass():
    from teleopit.sim.g1_o6.posture_report import summarize_posture

    result = summarize_posture(
        [{"phase": "INITIALIZE", "posture": {}, "contacts": []}], 0.02
    )
    assert result["max_hold_wrist_bend_deg"] is None
    assert result["min_hold_housing_clearance_m"] is None
    assert not result["hold_thumb_below_fingers_above"]


def test_report_requires_thumb_below_and_every_finger_above():
    from teleopit.sim.g1_o6.posture_report import ordered_contacts

    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    rows = [
        {
            "geoms": [
                "left_ladder_rung_11",
                "left_o6_lh_" + name + "_distal_collision",
            ],
            "regions": ["environment", "left_hand"],
            "normal_force": 1.0,
            "position": [0, 0, 1.0 if name == "thumb" else 3.0],
        }
        for name in fingers
    ]
    assert ordered_contacts(rows, "left", "left_ladder_rung_11", 2.0)
    assert not ordered_contacts(rows[:-1], "left", "left_ladder_rung_11", 2.0)
    rows[0]["position"][2] = 3.0
    assert not ordered_contacts(rows, "left", "left_ladder_rung_11", 2.0)


def test_report_counts_wrist_housing_even_below_support_gate():
    from teleopit.sim.g1_o6.posture_report import housing_contact

    contact = {
        "regions": ["environment", "other_robot"],
        "bodies": ["left_ladder_body", "left_wrist_roll_link"],
        "normal_force": 0.2,
    }
    assert housing_contact({"contacts": [contact]})
    contact["bodies"][1] = "left_o6_lh_index_distal"
    assert not housing_contact({"contacts": [contact]})


def test_unknown_posture_is_rejected_before_running(tmp_path):
    import pytest
    from teleopit.sim.g1_o6.config import Config
    from teleopit.sim.g1_o6.run import run_trial

    with pytest.raises(ValueError, match="unknown posture"):
        run_trial(Config(), tmp_path, posture="typo")
