"""Reports never replace a missing matched interval with an early-failure mean."""

from scripts.g1_o6_ladder_report import matched_statistics


def sample(time):
    return {
        "phase": "HOLD",
        "hold_s": time,
        "support_Fz_N": {"left_foot": 160, "right_foot": 162},
        "hands": {"left": {"share": 0.02}, "right": {"share": 0.015}},
        "base_reset_m": 0.01,
    }


def test_early_failure_has_no_matched_mean():
    result = matched_statistics([sample(20), sample(25)], 30, 25)
    assert result["status"] == "NOT RUN"
    assert "mean_support_Fz_N" not in result


def test_duration_windows_are_not_substituted_for_each_other():
    rows = [sample(t) for t in (20, 30, 50, 60, 90, 120)]
    for seconds, start in ((30, 20), (60, 50), (120, 90)):
        result = matched_statistics(rows, seconds, seconds)
        assert result["window_s"] == [start, seconds]
        assert result["samples"] == 2
        assert result["mean_hand_load_fraction"]["left"] == 0.02
    assert matched_statistics(rows[:2], 120, 120)["status"] == "NOT RUN"


def test_material_comparison_flags_hand_unloading_and_requires_both_windows():
    from scripts.g1_o6_ladder_report import compare_pair

    first = {
        "matched": matched_statistics([sample(20), sample(30)], 30, 30),
        "matched_foot_slip": {},
    }
    unloaded = {
        "matched": {
            **first["matched"],
            "mean_hand_load_fraction": {"left": 0.001, "right": 0.015},
        },
        "matched_foot_slip": {},
    }
    result = compare_pair([first, unloaded])
    assert result["hand_load_change_over_50_percent"] == {"left": True, "right": False}
    assert compare_pair([first, None]) == {"status": "NOT RUN"}


def test_analysis_cache_requires_identical_inputs_and_plot_bytes(tmp_path):
    import json

    from scripts.g1_o6_ladder_report import cached_result
    from teleopit.sim.g1_o6_support.ladder_campaign import file_hash

    cache = tmp_path / "cache.json"
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"recorded plot bytes")
    identity = {"run": "immutable input", "analysis": "version"}
    cache.write_text(
        json.dumps(
            {
                "identity": identity,
                "result": {"status": "example"},
                "plot_sha256": {plot.name: file_hash(plot)},
            }
        )
    )
    assert cached_result(cache, identity, tmp_path) == {"status": "example"}
    assert cached_result(cache, {**identity, "analysis": "changed"}, tmp_path) is None
    plot.write_bytes(b"changed plot bytes")
    assert cached_result(cache, identity, tmp_path) is None
