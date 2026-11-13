"""Campaign budgets, exact identities and promotion gates are behavioral contracts."""

from copy import deepcopy


def test_screen_matrix_has_exactly_25_unique_conditions():
    from teleopit.sim.g1_o6_support.ladder_campaign import conditions

    rows = conditions()
    assert len(rows) == len({r["id"] for r in rows}) == 25
    assert sum(r["stage_group"] == "A" for r in rows) == 7
    assert sum(r["stage_group"] == "B" for r in rows) == 12
    assert sum(r["stage_group"] == "C" for r in rows) == 6
    assert {r["geometry"] for r in rows if r["stage_group"] == "B"} == {
        "G0",
        "G5",
        "G6",
    }


def test_reset_shortlist_is_bounded_deterministic_and_preserves_controls():
    from teleopit.sim.g1_o6_support.ladder_reset import candidate_pose, reset_candidates
    from teleopit.sim.ladder_config import geometry_matrix

    for identity, geometry in geometry_matrix().items():
        rows = reset_candidates(geometry)
        assert 1 <= len(rows) <= 12
        assert rows == reset_candidates(geometry)
        assert len({(r["hand"], r["foot"]) for r in rows}) == len(rows)
        pose = candidate_pose(rows[0])
        assert pose.hand_rungs[0] == pose.hand_rungs[1]
        assert pose.foot_rungs[0] == pose.foot_rungs[1]
        assert pose.closed_pose == (0.05, 0.5, 1.35, 1.35, 1.35, 1.35)
        assert pose.finger_preload_Nm == 0.12
        if identity in ("G0", "G1", "G2", "G3"):
            assert (rows[0]["hand"], rows[0]["foot"]) == (11, 4)
        if identity == "G6":
            assert (rows[0]["hand"], rows[0]["foot"]) == (8, 3)


def passing_result():
    return {
        "strict_status": "PASS",
        "contact_quality": {"status": "PASS"},
        "physics_audit": {
            "no_external_support": True,
            "unintended_support_steps": 0,
            "housing_contact_steps": 0,
        },
        "drift_assessment": {"status": "FAIL", "windows": []},
    }


def test_promotions_reject_failed_opposition_assistance_and_prohibited_support():
    from teleopit.sim.g1_o6_support.ladder_campaign import screen_passes

    good = passing_result()
    assert screen_passes(good)
    for change in (("strict_status", "FAIL"), ("contact_quality", {"status": "FAIL"})):
        bad = dict(good, **{change[0]: change[1]})
        assert not screen_passes(bad)
    for key, value in (
        ("no_external_support", False),
        ("unintended_support_steps", 1),
        ("housing_contact_steps", 1),
    ):
        bad = deepcopy(good)
        bad["physics_audit"][key] = value
        assert not screen_passes(bad)
    assert not screen_passes({})


def test_identity_distinguishes_every_experiment_boundary():
    from teleopit.sim.g1_o6_support.ladder_campaign import run_identity

    baseline = {
        "condition": {"geometry": "G0", "material": {"scope": "H", "mu": 1.8}},
        "reset": {"pose": {"hand_rungs": [11, 11]}},
        "seconds": 30,
        "dt": 0.001,
        "provenance": {"code": "abc", "assets": "def", "controller": "original"},
    }
    initial = run_identity(**baseline)
    for key, value in (
        ("seconds", 60),
        ("dt", 0.0005),
        ("condition", {"geometry": "G6", "material": {"scope": "M", "mu": 0.4}}),
        ("reset", {"pose": {"hand_rungs": [8, 8]}}),
        ("provenance", {"code": "changed", "assets": "def", "controller": "original"}),
    ):
        assert run_identity(**dict(baseline, **{key: value})) != initial


def test_missing_final_window_is_not_a_stability_pass():
    from teleopit.sim.g1_o6_support.ladder_campaign import condition_status

    assert condition_status([]) == "NOT RUN"
    assert (
        condition_status(
            [{"stage": "screen", "result": {"reset_status": "INFEASIBLE"}}]
        )
        == "INFEASIBLE"
    )
    assert (
        condition_status([{"stage": "120s", "result": passing_result()}])
        == "duration passed, stability not qualified"
    )
