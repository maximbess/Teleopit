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


def test_representatives_show_later_failure_instead_of_only_the_short_pass():
    from scripts.g1_o6_pose_report import representative_runs

    def result(name, stage, kind, strict, seconds):
        return {
            "candidate": name,
            "stage": stage,
            "configuration": {"kind": kind},
            "strict": strict,
            "requested_s": seconds,
            "result": {"contact_quality": {"status": strict}},
        }

    short = result("freeze_all", "60s", "controller", "PASS", 60)
    reference = result("reference", "60s", "geometry", "PASS", 60)
    geometry = result("spacing16", "60s", "geometry", "PASS", 60)
    baseline = result("reference", "120s", "geometry", "FAIL", 120)
    capped = result("cap_thumb", "120s", "controller", "FAIL", 120)
    failed = result("freeze_all", "120s", "controller", "FAIL", 120)
    rows = [
        short,
        reference,
        geometry,
        result("reference", "screen", "geometry", "PASS", 30),
        baseline,
        capped,
        failed,
        result("rung10", "screen", "geometry", "FAIL", 30),
    ]
    chosen = representative_runs(rows)
    assert chosen["baseline"] is baseline
    assert chosen["best_geometry"] is geometry
    assert chosen["best_controller"] is failed
    assert chosen["final_qualification_or_best_failure"] is capped
