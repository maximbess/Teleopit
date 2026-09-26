"""Tests for the ladder environment-count throughput sweep."""

from __future__ import annotations

import io

import pytest

from train_mimic.scripts.benchmark_ladder_envs import (
    _OverflowTee,
    choose_num_envs,
    classify_failure,
    device_index,
    parse_gpu_line,
    rss_gib,
    summarize_run,
    validate_batch,
)

_GPU_FIELDS = (
    "sm_util",
    "mem_util",
    "mem_used_mib",
    "mem_total_mib",
    "sm_clock_mhz",
    "sm_clock_max_mhz",
    "power_w",
    "sw_power_cap",
    "hw_thermal",
)


def _sample(
    *,
    sm: float,
    mem: float,
    used: float,
    power_cap: bool = False,
    thermal: bool = False,
) -> dict[str, object]:
    return {
        "sm_util": sm,
        "mem_util": mem,
        "mem_used_mib": used,
        "mem_total_mib": 24576.0,
        "sm_clock_mhz": 1800.0,
        "sm_clock_max_mhz": 2000.0,
        "power_w": 250.0,
        "sw_power_cap": power_cap,
        "hw_thermal": thermal,
    }


def test_parse_gpu_line_reads_util_clocks_and_throttle_flags() -> None:
    parsed = parse_gpu_line(
        "81, 40, 8192, 24576, 1800, 2100, 260.5, Not Active, Active",
        _GPU_FIELDS,
    )

    assert parsed is not None
    assert parsed["sm_util"] == 81
    assert parsed["mem_util"] == 40
    assert parsed["mem_used_mib"] == 8192
    assert parsed["sm_clock_mhz"] == 1800
    assert parsed["power_w"] == 260.5
    assert parsed["sw_power_cap"] is False
    assert parsed["hw_thermal"] is True


def test_parse_gpu_line_accepts_missing_power_and_rejects_bad_rows() -> None:
    parsed = parse_gpu_line(
        "10, 0, 100, 24576, 500, 2000, [N/A]",
        _GPU_FIELDS[:7],
    )

    assert parsed is not None
    assert parsed["power_w"] is None
    assert parse_gpu_line("not,a,sample", _GPU_FIELDS[:7]) is None
    assert parse_gpu_line("1, 2", _GPU_FIELDS[:7]) is None


def test_rss_gib_uses_platform_units() -> None:
    assert rss_gib(1024**3, "darwin") == 1
    assert rss_gib(1024**2, "linux") == 1


def test_classify_failure_marks_memory_and_nan() -> None:
    assert classify_failure(RuntimeError("CUDA out of memory")) == "oom"
    assert classify_failure(RuntimeError("NaN in observation")) == "nan"
    assert classify_failure(RuntimeError("scene failed")) == "failed"


def test_validate_batch_requires_a_divisible_minibatch() -> None:
    validate_batch(128, 24, 4)
    with pytest.raises(ValueError, match="num_mini_batches"):
        validate_batch(1, 5, 4)


def test_device_index() -> None:
    assert device_index("cpu") is None
    assert device_index("cuda") == 0
    assert device_index("cuda:1") == 1


def test_summarize_run_drops_warmup_and_splits_gpu_phases() -> None:
    iterations = [
        {
            "collect_s": 9.0,
            "learn_s": 9.0,
            "collect_window": (0.0, 1.0),
            "learn_window": (1.0, 2.0),
        },
        {
            "collect_s": 2.0,
            "learn_s": 1.0,
            "collect_window": (10.0, 12.0),
            "learn_window": (12.0, 13.0),
        },
        {
            "collect_s": 4.0,
            "learn_s": 2.0,
            "collect_window": (20.0, 24.0),
            "learn_window": (24.0, 26.0),
        },
    ]
    samples = [
        (0.5, _sample(sm=99, mem=99, used=9)),
        (10.5, _sample(sm=10, mem=1, used=1000)),
        (12.0, _sample(sm=80, mem=40, used=2000, power_cap=True)),
        (22.0, _sample(sm=30, mem=5, used=1500)),
        (25.0, _sample(sm=40, mem=50, used=4000, thermal=True)),
        (26.0, _sample(sm=1, mem=1, used=9999)),
    ]

    row = summarize_run(
        num_envs=8,
        device="cuda:0",
        num_steps_per_env=24,
        num_mini_batches=4,
        startup_s=3.25,
        warmup=1,
        iterations=iterations,
        gpu_samples=samples,
        cpu_cores=1.5,
        host_rss_gib=2.5,
        torch_peak_bytes=3 * 1024**3,
        overflow_warnings=2,
        gpu_monitor="ok",
    )

    assert row["collect_s"] == 3
    assert row["learn_s"] == 1.5
    assert row["iteration_s"] == 4.5
    assert row["samples_per_s"] == round(192 / 4.5, 1)
    assert row["updates_per_hour"] == round(3600 / 4.5, 1)
    assert row["batch_size"] == 192
    assert row["minibatch_size"] == 48
    assert row["gpu_sm_util_collect_pct"] == 20
    assert row["gpu_sm_util_learn_pct"] == 60
    assert row["gpu_mem_bandwidth_util_collect_pct"] == 3
    assert row["gpu_mem_used_peak_mib"] == 4000
    assert row["gpu_sm_clock_frac_collect"] == 0.9
    assert row["gpu_throttle_fraction"] == 0.5
    assert row["gpu_samples_collect"] == 2
    assert row["gpu_samples_learn"] == 2
    assert row["overflow_warnings"] == 2
    assert row["torch_peak_gib"] == 3
    assert row["startup_s"] == 3.25


def test_choose_num_envs_prefers_the_smaller_near_peak_batch() -> None:
    rows = [
        {"num_envs": 128, "status": "ok", "samples_per_s": 90.0},
        {"num_envs": 256, "status": "ok", "samples_per_s": 100.0},
        {"num_envs": 512, "status": "ok", "samples_per_s": 50.0},
        {"num_envs": 1024, "status": "oom", "samples_per_s": None},
    ]

    assert choose_num_envs(rows) == 128
    assert choose_num_envs([{"status": "oom"}]) is None


def test_overflow_tee_keeps_constraint_warnings_across_split_writes() -> None:
    underlying = io.StringIO()
    matches: list[str] = []
    tee = _OverflowTee(underlying, matches)

    tee.write("step ok\nnefc overflow - please increase nj")
    tee.write("max\n")
    tee.write("nconmax too small\n")

    assert matches == [
        "nefc overflow - please increase njmax",
        "nconmax too small",
    ]
    assert "step ok" in underlying.getvalue()
