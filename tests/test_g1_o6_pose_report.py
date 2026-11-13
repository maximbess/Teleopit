"""Plots expose anatomical contact interruptions even when strict HOLD passes."""

import json


def test_contact_interruptions_are_visible_without_a_strict_threshold_crossing(
    tmp_path,
):
    from scripts.g1_o6_pose_report import failure_times

    run = tmp_path / "run"
    run.mkdir()
    (run / "all_phase_events.json").write_text(json.dumps([]))
    result = {
        "path": "run",
        "result": {
            "strict": {"events": [{"next": "HOLD", "time": 5.0}]},
            "physics_audit": {"first_crossings": {}},
        },
        "contact_first_absence": {
            "left:thumb_opposition": {
                "time_s": 7.0,
                "duration_s": 0.006,
                "followed_recorded_presence": True,
            },
            "right:valid": {
                "time_s": 5.001,
                "duration_s": 0.1,
                "followed_recorded_presence": False,
            },
            "left:housing:loaded_groups": {
                "time_s": 5.001,
                "duration_s": 30,
                "followed_recorded_presence": False,
            },
        },
    }
    events = failure_times(tmp_path, result)
    assert ("left:thumb_opposition absent for 0.006 s", 2.0) in events
    assert any("right:valid" in name and "initial" in name for name, _ in events)
    assert all("housing" not in name for name, _ in events)
