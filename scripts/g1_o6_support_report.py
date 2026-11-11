"""Build compact reports from preserved support-validation logs; no simulation."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def coupled_trace(path):
    initial = read(path / "resolved.json")["initial"]
    samples = [r for r in rows(path / "steps.jsonl") if r["phase"] == "HOLD"]
    points = np.array([r["base_pose"][:3] for r in samples])
    time = np.array([r["hold_time_s"] for r in samples])
    speed = np.linalg.norm(np.gradient(points, time, axis=0), axis=1) * 1000
    return {
        "time": time,
        "base_mm": [r["base_drift_m"] * 1000 for r in samples],
        "com_mm": np.linalg.norm(
            np.array([r["com"] for r in samples]) - initial["com"], axis=1
        )
        * 1000,
        "speed_mm_s": np.convolve(speed, np.ones(50) / 50, "valid"),
        "hand_share": [
            100 * r["support_measurement"]["total_hand_share"] for r in samples
        ],
        "palm_mm": palm_translation(samples),
    }


def palm_translation(samples):
    return {
        side: np.linalg.norm(
            np.array([r["hand_in_target_rung"][side]["position"] for r in samples])
            - samples[0]["hand_in_target_rung"][side]["position"],
            axis=1,
        )
        * 1000
        for side in ("left", "right")
    }


def save_figure(figure, output, name):
    figure.savefig(output / (name + ".png"), dpi=160)
    figure.savefig(output / (name + ".pdf"))
    plt.close(figure)


def long_plot(root, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for name, label, color in (
        ("hold_120s", "Original grasp", "#3b6c9e"),
        ("correction_120s", "Thumb target 0.30 rad", "#bf642d"),
    ):
        trace = coupled_trace(root / name)
        t = trace["time"]
        axes[0, 0].plot(t, trace["base_mm"], color=color, label=label + ": base")
        axes[0, 0].plot(t, trace["com_mm"], color=color, ls="--", label=label + ": COM")
        axes[0, 1].plot(t[25:-24], trace["speed_mm_s"], color=color, label=label)
        for side, style in (("left", "-"), ("right", "--")):
            axes[1, 0].plot(
                t,
                trace["palm_mm"][side],
                color=color,
                ls=style,
                label=label + ": " + side,
            )
        axes[1, 1].plot(t, trace["hand_share"], color=color, label=label)
    axes[0, 0].axhline(30, color="black", ls=":", label="Base strict limit")
    for ax, ylabel in zip(
        axes.ravel(),
        (
            "RESET translation (mm)",
            "Base speed, 1 s average (mm/s)",
            "Palm/rung translation since HOLD entry (mm)",
            "Signed total hand vertical share (%)",
        ),
    ):
        ax.set(xlabel="Scored HOLD time (s)", ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "120 s attempts: original drift failure; corrected grasp opposition failure\nRepeats and fine-step long qualification NOT RUN after preceding failure",
        fontsize=12,
    )
    save_figure(fig, output, "long_hold")


def capacity_paths(root, side):
    return [
        ("initial", root / "capacity_initial" / (side + "_zero")),
        ("settled", root / "capacity_settled" / (side + "_zero")),
        ("repeat", root / "capacity_repeat" / side / "zero"),
        ("fine", root / "capacity_fine" / side / "zero"),
    ]


def capacity_trace(path, side):
    scored = [
        r
        for r in rows(path / "steps.jsonl")
        if r.get("phase") in ("SETTLE", "LOAD_RAMP", "PLATEAU", "RAMP_DOWN", "RECOVERY")
    ]
    return {
        "time": np.array([r["time_s"] for r in scored]) - 6,
        "translation_mm": [1000 * r["from_hold"]["translation_m"] for r in scored],
        "slip_mm_s": [
            1000 * (r["supports"][side + "_hand"]["slip_m_s"] or 0) for r in scored
        ],
    }


def capacity_plot(root, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    response, response_axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for i, side in enumerate(("left", "right")):
        for label, path in capacity_paths(root, side):
            summary = read(path / "summary.json")
            trace = capacity_trace(path, side)
            axes[i, 0].plot(trace["time"], trace["translation_mm"], label=label)
            axes[i, 1].plot(trace["time"], trace["slip_mm_s"], label=label)
            response_axes[i].scatter(
                summary["achieved_force_norm_N"],
                summary["unsupported_elapsed_s"],
                marker="x",
                s=60,
                label=f"{label}: FAIL; plateau {summary['plateau_achieved_s']:g} s",
            )
        axes[i, 0].set(ylabel=side + " hand translation (mm)")
        axes[i, 1].set(ylabel=side + " contact slip (mm/s)")
        response_axes[i].set(
            title=side + " free wrist",
            xlabel="Achieved applied test force (N)",
            ylabel="Time until abort after fixture removal (s)",
            xlim=(-0.5, 2),
        )
        response_axes[i].text(
            0.65,
            0.55,
            "Nonzero loads / replay\nBLOCKED by settling failure",
            ha="center",
            transform=response_axes[i].transAxes,
            fontsize=8,
        )
        response_axes[i].legend(fontsize=8)
    for ax in axes.ravel():
        ax.set(xlabel="Time since fixture removal (s)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Both authentic isolated hands: gravity only, original grasp\nStrict failures; no completed loaded plateau or capacity reserve"
    )
    save_figure(fig, output, "isolated_traces")
    save_figure(response, output, "capacity_response")


def long_result(root, name):
    result = read(root / name / "summary.json")
    return {
        "run": name,
        "strict": result["strict_status"],
        "achieved_hold_s": result["strict"]["unsupported_hold_seconds"],
        "max_reset_drift_mm": 1000 * result["maximum_reset_drift_m"],
        "final_reset_drift_mm": 1000 * result["final_reset_drift_m"],
        "failures": result["strict"]["failure_reasons"],
        "drift_assessment": result["drift_assessment"],
        "physics_audit": result["physics_audit"],
    }


def report(root, output):
    output.mkdir(parents=True, exist_ok=True)
    names = ("control_before", "control_after", "control_final")
    checksums = {
        name: {f: digest(root / name / f) for f in ("steps.jsonl", "summary.json")}
        for name in names
    }
    value = {
        "schema": "g1_o6_support_report_v1",
        "control_sha256": checksums,
        "default_control_byte_identical": checksums[names[0]]
        == checksums[names[1]]
        == checksums[names[2]],
        "long_holds": [
            long_result(root, n)
            for n in (
                "hold_60s",
                "hold_120s",
                "correction_30s",
                "correction_60s",
                "correction_120s",
            )
        ],
        "capacity": {
            side: {
                label: read(path / "summary.json")
                for label, path in capacity_paths(root, side)
            }
            for side in ("left", "right")
        },
        "later_physical_stages": {
            mode: "BLOCKED" for mode in ("load", "unload", "release", "reacquire")
        },
        "free_closing": read(root / "free_close_final" / "free_close_results.json"),
    }
    (output / "results.json").write_text(json.dumps(value, indent=2) + "\n")
    long_plot(root, output)
    capacity_plot(root, output)
    print(
        json.dumps(
            {
                "control_byte_identical": value["default_control_byte_identical"],
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=Path("outputs/g1_o6/support_transfer")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/g1_o6/support_transfer")
    )
    args = parser.parse_args()
    report(args.runs, args.output)
