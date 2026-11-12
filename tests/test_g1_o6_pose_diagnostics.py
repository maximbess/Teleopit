import copy

import numpy as np


def contact(group, normal, z, force=1.0):
    return {
        "group": group,
        "target": True,
        "normal_force_N": force,
        "normal_on_hand_world": normal,
        "position_world_m": [0, 0, z],
        "rung_position_m": [0, 0, z],
    }


def test_thumb_specific_opposition_cannot_be_replaced_by_two_other_fingers():
    from teleopit.sim.g1_o6_support.pose_contacts import grasp_quality

    fingers = [
        contact(n, [0, 0, 1], 0.01) for n in ("index", "middle", "ring", "pinky")
    ]
    fingers[1]["normal_on_hand_world"] = [0, 0, -1]
    assert not grasp_quality(fingers, 0)["thumb_opposition"]
    thumb = contact("thumb", [0, 0, -1], -0.01)
    result = grasp_quality([thumb, *fingers], 0)
    assert result["thumb_opposition"] and result["ordered"]
    thumb["normal_force_N"] = 0.049
    assert not grasp_quality([thumb, *fingers], 0)["valid"]
    assert grasp_quality([thumb, *fingers], 0)["geometric_groups"]["thumb"]


def test_friction_reports_separate_translational_and_torsional_utilization():
    from teleopit.sim.g1_o6_support.pose_contacts import friction_usage

    result = friction_usage([10, 3, 4, 0.2, 0, 0], [1, 2, 0.1, 0.01, 0.01], 4)
    assert result["translation_only_utilization"] == np.sqrt(0.3**2 + 0.2**2)
    assert result["torsional_utilization"] == 0.2
    assert friction_usage([0] * 6, [1] * 5, 3)["torsional_utilization"] is None


def test_contact_episodes_track_groups_instead_of_solver_ids():
    from teleopit.sim.g1_o6_support.pose_contacts import ContactEpisodes

    history = ContactEpisodes(0.001)
    history.update(1, {"left:thumb:loaded": True})
    history.update(1.001, {"left:thumb:loaded": True})
    history.update(1.002, {"left:thumb:loaded": False})
    history.update(1.501, {"left:thumb:loaded": False})
    assert len(history.rows) == 2
    np.testing.assert_allclose(history.rows[0]["duration_s"], 0.002)
    assert history.longest_absence()["left:thumb:loaded"] >= 0.5 - 1e-9


def test_extended_observer_does_not_write_simulation_state(tmp_path):
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6_support.pose_observer import PoseObserver

    env = Environment()
    before = {
        n: getattr(env.data, n).copy()
        for n in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")
    }
    observer = PoseObserver(env, tmp_path)
    observer(env)
    row = observer.control_sample(env)
    observer.close()
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(env.data, name), expected)
    assert set(row["hands"]) == {"left", "right"}
    assert all(
        "left_ladder_rung_11" in hand["clearance_by_obstacle_m"]
        for hand in row["hands"].values()
    )


def test_experimental_controller_sources_are_explicitly_rejected():
    import pytest

    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6.metadata import resolved
    from teleopit.sim.o6_diagnostics.source import validate_contract

    env = Environment()
    meta = resolved(env, snapshot(env))
    meta = copy.deepcopy(meta)
    meta["controller"]["schema"] = "o6_hold_reference_v1"
    with pytest.raises(ValueError, match="unsupported source controller"):
        validate_contract(meta)
