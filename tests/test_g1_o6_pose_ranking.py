def test_unloaded_hands_are_flagged_per_side_without_absolute_force_substitution():
    from teleopit.sim.g1_o6_support.pose_results import load_flag

    assert load_flag(
        {"left": -0.02, "right": 0.012}, {"left": 0.01, "right": 0.01}
    ) == {"left": True, "right": False}
    assert load_flag(None, {"left": 0.01}) is None


def test_quality_failure_cannot_be_a_finalist():
    from teleopit.sim.g1_o6_support.pose_selection import finalists

    base = {
        "candidate": "example",
        "stage": "120s",
        "strict": "PASS",
        "result": {
            "progression_justified": False,
            "contact_quality": {"status": "FAIL"},
        },
    }
    assert finalists([base]) == []
    base["result"]["progression_justified"] = True
    base["result"]["contact_quality"]["status"] = "PASS"
    assert finalists([base]) == ["example"]


def test_combination_requires_two_independently_supported_nonconflicting_changes():
    from teleopit.sim.g1_o6_support.pose_candidates import initial_geometry
    from teleopit.sim.g1_o6_support.pose_selection import combine_improvements

    candidates = {c.id: c for c in initial_geometry()}
    parent = candidates["reference"]
    children = [
        c
        for c in candidates.values()
        if c.parent == parent.id and c.family in ("hand_tilt_deg", "waist_lean_rad")
    ]
    selected = combine_improvements(parent, [children[0], children[2]])
    assert len(selected) == 1 and len(selected[0].changes) == 2
    assert combine_improvements(parent, children[:2]) == []
