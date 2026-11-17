"""Thumb evidence comes from native contacts, not a requested finger target."""

from copy import deepcopy
import pytest


def contact(group, z, normal, force=1.):
    return {"bodies": ["ladder", "left_o6_lh_" + group + "_distal"],
            "regions": ["environment", "left_hand"],
            "geoms": ["left_ladder_rung_07", "left_o6_" + group],
            "normal_force": force, "normal_on_second_world": normal,
            "position": [0, 0, z]}


def test_raw_contact_reconstruction_requires_real_ordered_opposition_on_intended_tread():
    from scripts.g1_o6_g6_acquisition_report import raw_hand

    row = {"support_targets": {"left_hand": "left_ladder_rung_07"},
           "contacts": [contact("thumb", -.02, [0, 0, -1]),
                        *[contact(g, .02, [0, 0, 1]) for g in ("index", "middle", "ring", "pinky")]]}
    measured = raw_hand(row, "left", 0)
    assert measured == {"valid": True, "thumb_normal_N": 1., "minimum_finger_normal_N": 1.}
    for field, value in (("normal_force", .04), ("position", [0, 0, .01]),
                         ("normal_on_second_world", [0, 0, 1]),
                         ("geoms", ["left_ladder_rung_08", "left_o6_thumb"])):
        wrong = deepcopy(row)
        wrong["contacts"][0][field] = value
        assert not raw_hand(wrong, "left", 0)["valid"]
    row["contacts"].pop()
    assert not raw_hand(row, "left", 0)["valid"]


def test_raw_contact_reconstruction_accounts_for_reversed_geom_order():
    from scripts.g1_o6_g6_acquisition_report import raw_hand

    contacts = [contact("thumb", -.02, [0, 0, -1]),
                *[contact(g, .02, [0, 0, 1]) for g in ("index", "middle", "ring", "pinky")]]
    for entry in contacts:
        for key in ("bodies", "regions", "geoms"):
            entry[key].reverse()
        entry["normal_on_second_world"] = [-v for v in entry["normal_on_second_world"]]
    measured = raw_hand({"contacts": contacts, "support_targets": {"left_hand": "left_ladder_rung_07"}}, "left", 0)
    assert measured["valid"] and measured["thumb_normal_N"] == 1.


def test_acquisition_verifier_rejects_a_ready_claim_without_a_continuous_contact_tail():
    from scripts.g1_o6_g6_acquisition_report import check_report

    values = {"samples": 500, "anatomical_samples": 0, "tail_samples": 0,
              "peak_thumb_normal_N": {"left": 0., "right": 0.}}
    expected = {**values, "valid_samples": 0}
    reported = {"phases": {"ACQUIRE": expected}, "ready_for_removal": True, "ready_for_hold": False}
    with pytest.raises(AssertionError):
        check_report({"ACQUIRE": values}, reported, .001)


def test_late_acquisition_readiness_is_checked_at_the_actual_hold_boundary():
    from scripts.g1_o6_g6_acquisition_report import check_report

    values = {"samples": 500, "anatomical_samples": 0, "tail_samples": 0,
              "peak_thumb_normal_N": {"left": 1., "right": 1.}}
    actual = {"ACQUIRE": values, "REMOVE_ASSISTANCE": {**values, "anatomical_samples": 500, "tail_samples": 500}}
    reported = {"phases": {phase: {**v, "valid_samples": v["anatomical_samples"]} for phase, v in actual.items()},
                "ready_for_removal": False, "ready_for_hold": True}
    check_report(actual, reported, .001)
