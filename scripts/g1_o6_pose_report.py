"""Compare the bounded pose campaign using preserved traces, never new physics."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from teleopit.sim.g1_o6_support.pose_candidates import Candidate
from teleopit.sim.g1_o6_support.pose_results import (
    campaign_results,
    read,
    series,
    strict_contact_pass,
)
from teleopit.sim.g1_o6_support.pose_validation import stage_qualification


def trajectory_change(rows, side, name):
    initial = next(
        (
            r["hands"][side][name]
            for r in rows
            if r["phase"] == "HOLD" and r["hands"][side][name] is not None
        ),
        None,
    )
    if initial is None:
        return [np.nan] * len(rows)
    return [
        1000 * np.linalg.norm(np.array(r["hands"][side][name]) - initial)
        if r["hands"][side][name] is not None
        else np.nan
        for r in rows
    ]


def hand_lines(axes, rows, side, color):
    time = [r["hold_s"] for r in rows]
    hand = [r["hands"][side] for r in rows]
    signals = (
        (1, "thumb_yaw_rad", "angle", "-", 1),
        (1, "thumb_target_rad", "applied target", "--", 1),
        (4, "thumb_normal_N", "thumb", "-", 1),
        (4, "finger_min_normal_N", "min finger", "--", 1),
        (5, "wrist_force_N", "|wrist F|", "-", 1),
        (6, "share", "share", "-", 100),
        (7, "wrist_bend_deg", "wrist", "-", 1),
        (7, "elbow_bend_deg", "elbow", "--", 1),
        (8, "wrist_moment_Nm", "|wrist M|", "-", 1),
        (9, "arm_effort_fraction", "peak arm effort", "-", 1),
    )
    for panel, field, label, style, scale in signals:
        axes[panel].plot(
            time,
            [scale * h[field] for h in hand],
            color=color,
            ls=style,
            label=side + " " + label,
        )
    axes[2].plot(
        time, trajectory_change(rows, side, "thumb_centroid_m"), color=color, label=side
    )
    axes[3].plot(
        time,
        trajectory_change(rows, side, "palm_position_m"),
        color=color,
        label=side + " palm",
    )
    axes[3].plot(
        time,
        trajectory_change(rows, side, "mount_position_m"),
        color=color,
        ls="--",
        label=side + " mount",
    )


def failure_times(root, result):
    start = next(
        (
            e["time"]
            for e in result["result"]["strict"]["events"]
            if e["next"] == "HOLD"
        ),
        0,
    )
    scored = result["result"]["physics_audit"]["first_crossings"]
    all_events = read(root / result["path"] / "all_phase_events.json")
    times = [(name, v["first_crossing_s"] - start) for name, v in scored.items()]
    for name in scored:
        previous = next((r for r in all_events if r["event"] == name), None)
        if (
            previous
            and previous["first_crossing_s"] < scored[name]["first_crossing_s"] - 0.02
        ):
            times.append((name + " before HOLD", previous["first_crossing_s"] - start))
    return times + contact_times(result, start)


def contact_times(result, start):
    times = []
    for name, event in result.get("contact_first_absence", {}).items():
        if ":thumb" in name or name.endswith((":ordered", ":valid")):
            label = f"{name} absent for {event['duration_s']:g} s"
            if not event["followed_recorded_presence"]:
                label += " (initial absence)"
            times.append((label, event["time_s"] - start))
    return times


def plot_run(root, result, output, label):
    rows = series(root / result["path"])
    figure, grid = plt.subplots(5, 2, figsize=(15, 17), sharex=True)
    axes = grid.flatten()
    time = [r["hold_s"] for r in rows]
    axes[0].plot(
        time, [1000 * r["base_reset_m"] for r in rows], label="base from RESET"
    )
    axes[0].plot(
        time,
        [
            1000 * r["base_hold_m"] if r["base_hold_m"] is not None else np.nan
            for r in rows
        ],
        label="base from HOLD",
    )
    axes[0].axhline(30, color="crimson", ls=":", label="unchanged strict limit")
    for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
        hand_lines(axes, rows, side, color)
    labels = (
        "Base displacement (mm)",
        "Thumb yaw (rad)",
        "Thumb contact-centroid movement (mm)",
        "Rung-relative movement (mm)",
        "Loaded group normal force (N)",
        "Parent-on-hand force at mount (N)",
        "Signed vertical hand share (%)",
        "Physical arm bends (degrees)",
        "Parent-on-hand moment at mount (Nm)",
        "Arm actuator effort / physical limit",
    )
    events = failure_times(root, result)
    for ax, title in zip(axes, labels):
        ax.set_ylabel(title)
        ax.axvline(0, color="black", lw=0.7)
        for _, moment in events:
            ax.axvline(moment, color="crimson", lw=0.7, alpha=0.5)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    axes[-1].set_xlabel("Time relative to HOLD entry (s)")
    axes[-2].set_xlabel("Time relative to HOLD entry (s)")
    figure.suptitle(
        f"{label}: {result['candidate']} / {result['stage']}\nStrict {result['strict']}; late motion {result['result']['drift_assessment']['status']}; contact {result['result']['contact_quality']['status']}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_views(figure, axes, output, label, time, events)
    plt.close(figure)


def save_views(figure, axes, output, label, time, events):
    figure.savefig(output / (label + ".png"), dpi=140)
    figure.savefig(output / (label + ".pdf"))
    windows = [("hold_entry", 0, "HOLD entry")] + [
        ("first_" + str(i), t, name) for i, (name, t) in enumerate(events)
    ]
    title = figure._suptitle.get_text()
    for name, moment, event in windows:
        figure.suptitle(
            title + f"\nWindow: {event} at HOLD {moment:.3f} s", fontsize=10
        )
        for ax in axes:
            ax.set_xlim(max(min(time), moment - 2), min(max(time), moment + 2))
        figure.savefig(output / (label + "_" + name + ".png"), dpi=140)
    (output / (label + "_events.json")).write_text(json.dumps(windows, indent=2) + "\n")


def representative_runs(results):
    by_id = {(r["candidate"], r["stage"]): r for r in results}
    selected = {
        "baseline": by_id.get(("reference", "120s"), by_id[("reference", "screen")]),
        "rung10": by_id[("rung10", "screen")],
    }
    for kind in ("geometry", "controller"):
        eligible = kind_results(results, kind)
        if eligible:
            selected["best_" + kind] = latest_attempt(results, eligible[0]["candidate"])
    longer = diagnostic_long_attempts(results)
    if longer:
        selected["final_qualification_or_best_failure"] = latest_attempt(
            results, longer[0]["candidate"]
        )
    return selected


def kind_results(results, kind):
    eligible = [
        r
        for r in results
        if r["configuration"]["kind"] == kind and strict_contact_pass(r)
    ]
    return alternatives(eligible)


def alternatives(results):
    # The reference already has its own panel. Keep it when no alternative exists.
    return [r for r in results if r["candidate"] != "reference"] or results


def diagnostic_long_attempts(results):
    return alternatives(
        [r for r in results if r["stage"] in ("120s", "repeat", "fine")]
    )


def latest_attempt(results, name):
    stage_order = {"repeat": 1, "fine": 2}
    return max(
        (r for r in results if r["candidate"] == name),
        key=lambda r: (r["requested_s"], stage_order.get(r["stage"], 0)),
    )


def qualification_table(state):
    result = {}
    for name, candidate in state["candidates"].items():
        stages = stage_qualification(
            state["runs"], name, Candidate.from_dict(candidate).digest()
        )
        result[name] = {
            "stages": stages,
            "qualified": all(stages.values()),
            "statuses": {
                stage: validation_status(state, name, stage)
                for stage in ("120s", "repeat", "fine")
            },
        }
    return result


def validation_status(state, name, stage):
    record = next(
        (
            r
            for r in reversed(state["runs"])
            if r["candidate"] == name and r["stage"] == stage
        ),
        None,
    )
    if record is None:
        return "NOT RUN"
    if record["status"] != "COMPLETE":
        return record["status"]
    return "PASS" if record["result"].get("progression_justified") else "FAIL"


def comparison_markdown(results, qualified):
    lines = [
        "# Bounded long-HOLD pose campaign",
        "",
        "This report is generated from saved results. A screen is not 120 s qualification.",
        "",
        "| Candidate / stage | Strict / late / contact | HOLD achieved (s) | Max RESET drift (mm) | Mean hand shares L/R (%) | First terminal reason |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(comparison_row(row))
    names = [n for n, value in qualified.items() if value["qualified"]]
    lines.extend(
        [
            "",
            "Qualified nominal/repeat/fine candidates: "
            + (", ".join(names) if names else "NONE")
            + ".",
            "",
            "[Complete configurations, per-hand metrics, late windows, first crossings and qualification status](results.json).",
            "[Flat comparison table with all per-hand metrics and separate validation statuses](comparison.csv).",
            "",
            "A >50% reduction in either hand's signed 20–30 s share is explicitly flagged in the JSON; it does not establish stronger grasp capacity.",
            "",
            "Contact quality requires all scored control samples to retain thumb-specific opposition and accepted ordering, with no physics-rate quality-loss episode lasting 0.5 s. Shorter interruptions remain reported.",
        ]
    )
    return "\n".join(lines) + "\n"


def comparison_row(row):
    result = row["result"]
    late = result.get("drift_assessment", {}).get("status", "NOT RUN")
    contact = result.get("contact_quality", {}).get("status", "NOT RUN")
    drift = result.get("maximum_reset_drift_m")
    drift_text = f"{1000 * drift:.3f}" if drift is not None else "—"
    loads = (
        "/".join(
            f"{100 * row['hands'][s]['mean_share']:.3f}" for s in ("left", "right")
        )
        if all(row["hands"].values()) and row["hands"]
        else "—"
    )
    reasons = result.get("strict", {}).get("failure_reasons", result.get("reasons", []))
    return f"| {row['candidate']} / {row['stage']} | {row['strict']} / {late} / {contact} | {row['achieved_s']:.2f} | {drift_text} | {loads} | {', '.join(reasons) or 'none'} |"


def flatten_fields(value, prefix=""):
    result = {}
    for name, item in value.items():
        key = prefix + str(name)
        if isinstance(item, dict):
            result.update(flatten_fields(item, key + "."))
        else:
            result[key] = json.dumps(item) if isinstance(item, (list, tuple)) else item
    return result


def comparison_csv(results, qualified, output):
    rows = [
        flatten_fields(
            {
                "candidate": r["candidate"],
                "parent": r["configuration"]["parent"],
                "stage": r["stage"],
                "requested_s": r["requested_s"],
                "achieved_s": r["achieved_s"],
                "changes": r["configuration"]["changes"],
                "pose": r["configuration"]["pose"],
                "ik_seed": r["configuration"]["ik_seed"],
                "controller": r["controller"],
                "targets": r["support_targets"],
                "reset_inspection": r["reset"],
                "hands": r["hands"],
                "support_accounting": r.get("support_accounting", {}),
                "result": r["result"],
                "late_final_window": final_window(r["result"]),
                "contact_first_absence": r.get("contact_first_absence", {}),
                "matched_20_30s_hand_shares": r["matched_shares"],
                "materially_reduced_hand_load": r["materially_reduced_hand_load"],
                "eligible_for_longer_diagnostic": r["stage"] in ("screen", "60s")
                and strict_contact_pass(r),
                "nominal_repeat_fine": qualified[r["candidate"]],
            }
        )
        for r in results
    ]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def final_window(result):
    windows = result.get("drift_assessment", {}).get("windows", [])
    return windows[-1] if windows else {}


def report(root, output, plots=True):
    state = read(root / "campaign_state.json")
    results = campaign_results(SimpleNamespace(root=root, state=state))
    qualified = qualification_table(state)
    representatives = representative_runs(results)
    output.mkdir(parents=True, exist_ok=True)
    configs = output / "candidates"
    configs.mkdir(exist_ok=True)
    for name, candidate in state["candidates"].items():
        (configs / (name + ".json")).write_text(json.dumps(candidate, indent=2) + "\n")
    record = {
        "schema": "g1_o6_pose_results_v1",
        "results": results,
        "qualification": qualified,
        "frozen_decisions": state["decisions"],
        "budgets": state["specification"]["budgets"],
        "representatives": {name: r["path"] for name, r in representatives.items()},
        "transfer_and_capacity": "BLOCKED unless separately requalified; pose search never enables transfer",
    }
    (output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "COMPARISON.md").write_text(
        comparison_markdown(results, qualified), encoding="utf-8"
    )
    comparison_csv(results, qualified, output)
    if plots:
        for name, result in representatives.items():
            if result["strict"] != "NOT RUN":
                plot_run(root, result, output, name)
    print(
        json.dumps(
            {
                "completed_trials": len(results),
                "qualified": [n for n, v in qualified.items() if v["qualified"]],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("outputs/g1_o6/pose_search"))
    parser.add_argument("--output", type=Path, default=Path("docs/g1_o6/pose_search"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    report(args.runs, args.output, not args.no_plots)
