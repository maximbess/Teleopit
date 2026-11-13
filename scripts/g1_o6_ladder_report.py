"""Matched-window ladder/material comparisons from immutable recorded evidence."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from teleopit.sim.g1_o6_support.ladder_campaign import condition_status, file_hash
from teleopit.sim.g1_o6_support.pose_results import hold_entry, series


def read(path):
    return json.loads(path.read_text())


def in_hold_window(phase, time, start, end):
    return phase == "HOLD" and time is not None and start - 1e-8 <= time <= end + 0.021


def complete_window(rows, start, end):
    return (
        bool(rows)
        and rows[0]["hold_s"] <= start + 0.021
        and rows[-1]["hold_s"] >= end - 0.021
    )


def matched_statistics(rows, requested, achieved):
    start = {30: 20, 60: 50, 120: 90}[requested]
    if achieved < requested - 0.021:
        return {
            "status": "NOT RUN",
            "window_s": [start, requested],
            "reason": "trial ended before matched window completed",
        }
    matched = [
        r for r in rows if in_hold_window(r["phase"], r["hold_s"], start, requested)
    ]
    if not complete_window(matched, start, requested):
        return {
            "status": "NOT RUN",
            "window_s": [start, requested],
            "reason": "complete recorded window absent",
        }
    return dict(
        status="MEASURED",
        window_s=[start, requested],
        samples=len(matched),
        **window_averages(matched),
    )


def window_averages(matched):
    supports = {
        name: float(np.mean([r["support_Fz_N"][name] for r in matched]))
        for name in matched[0]["support_Fz_N"]
    }
    return {
        "mean_support_Fz_N": supports,
        "mean_hand_load_fraction": {
            side: float(np.mean([r["hands"][side]["share"] for r in matched]))
            for side in ("left", "right")
        },
        "mean_reset_drift_m": float(np.mean([r["base_reset_m"] for r in matched])),
        "max_reset_drift_m": max(r["base_reset_m"] for r in matched),
    }


def matched_foot_slip(path, requested, achieved):
    if achieved < requested - 0.021:
        return None
    start = {30: 20, 60: 50, 120: 90}[requested]
    slips = {side: [] for side in ("left", "right")}
    with (path / "steps.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            time = row.get("hold_time_s")
            if in_hold_window(row["phase"], time, start, requested):
                for side, values in slips.items():
                    values.append(row["supports"][side + "_foot"]["max_slip"])
    return {
        side: {"mean_m_s": float(np.mean(values)), "max_m_s": max(values)}
        for side, values in slips.items()
        if values
    }


def summarize(root, record):
    path = root / record["output"]
    summary = read(path / "summary.json")
    requested = {"screen": 30, "60s": 60, "120s": 120, "repeat": 120, "fine": 120}[
        record["stage"]
    ]
    achieved = summary.get("strict", {}).get("unsupported_hold_seconds", 0)
    result = {
        "condition": record["candidate"],
        "stage": record["stage"],
        "path": record["output"],
        "requested_s": requested,
        "achieved_s": achieved,
        "strict": summary["strict_status"],
        "contact": summary.get("contact_quality", {}).get("status", "NOT RUN"),
        "late_motion": summary.get("drift_assessment", {}).get("status", "NOT RUN"),
        "terminal_cause": summary.get("strict", {}).get(
            "failure_reasons", summary.get("reasons", [])
        ),
        "maximum_reset_drift_m": summary.get("maximum_reset_drift_m"),
        "first_crossings": summary.get("physics_audit", {}).get("first_crossings", {}),
        "first_persistent": summary.get("physics_audit", {}).get(
            "first_persistent", {}
        ),
        "summary": summary,
    }
    event_path = path / "all_phase_events.json"
    result["all_phase_events"] = read(event_path) if event_path.exists() else []
    if (path / "steps.jsonl").exists():
        metadata = read(path / "resolved.json")
        result["surface_displacement_from_G0_m"] = metadata.get("ladder", {}).get(
            "surface_displacement_from_G0_m"
        )
        rows = series(path)
        result["matched"] = matched_statistics(rows, requested, achieved)
        result["matched_foot_slip"] = matched_foot_slip(path, requested, achieved)
        result["contact_migration_m"] = summary.get("contact_quality", {}).get(
            "max_contact_centroid_migration_m", {}
        )
        return result, rows
    result["matched"] = {"status": "NOT RUN"}
    return result, []


def plot_run(rows, result, destination):
    if not rows:
        return
    time = [r["hold_s"] for r in rows]
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    axes = axes.ravel()
    axes[0].plot(time, [1000 * r["base_reset_m"] for r in rows], label="from RESET")
    axes[0].plot(
        time, [1000 * (r["base_hold_m"] or 0) for r in rows], label="from HOLD"
    )
    for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
        plot_side(axes, rows, time, side, color)
    format_axes(axes)
    crossings = failure_crossings(result)
    for axis in axes:
        for crossing in crossings:
            axis.axvline(crossing, color="crimson", lw=0.7, alpha=0.6)
    save_plots(figure, axes, time, result, crossings, destination)


def plot_side(axes, rows, time, side, color):
    hands = [r["hands"][side] for r in rows]
    for field, label, style in (
        ("thumb_normal_N", "thumb", "-"),
        ("finger_min_normal_N", "min finger", "--"),
    ):
        axes[1].plot(
            time,
            [h[field] for h in hands],
            color=color,
            ls=style,
            label=side + " " + label,
        )
    axes[2].plot(
        time,
        [int(h["valid"]) for h in hands],
        color=color,
        label=side + " anatomical gate",
    )
    plot_migration(axes[3], rows, time, side, color)
    axes[4].plot(time, [100 * h["share"] for h in hands], color=color, label=side)
    axes[5].plot(
        time,
        [r["support_Fz_N"][side + "_foot"] for r in rows],
        color=color,
        label=side + " foot",
    )


def plot_migration(axis, rows, time, side, color):
    hands = [r["hands"][side] for r in rows]
    initial = next((r["hands"][side] for r in rows if r["phase"] == "HOLD"), hands[0])
    for label, style in (("palm_position_m", "-"), ("mount_position_m", "--")):
        axis.plot(
            time,
            [1000 * np.linalg.norm(np.array(h[label]) - initial[label]) for h in hands],
            color=color,
            ls=style,
            label=side + " " + label.split("_")[0],
        )


def format_axes(axes):
    labels = (
        "Base drift (mm)",
        "Group normal force (N)",
        "Thumb opposition + ordering",
        "Rung-relative migration (mm)",
        "Signed hand load (% weight)",
        "Signed foot Fz (N)",
    )
    axes[2].set_ylim(-0.05, 1.05)
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.axvline(0, color="black", lw=0.7)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    for axis in axes[-2:]:
        axis.set_xlabel("Time relative to HOLD entry (s)")


def failure_crossings(result):
    entry = hold_entry(result["summary"])
    return (
        [v["first_crossing_s"] - entry for v in result["first_crossings"].values()]
        if entry is not None
        else []
    )


def save_plots(figure, axes, time, result, crossings, destination):
    figure.suptitle(
        f"{result['condition']} / {result['stage']} | HOLD {result['achieved_s']:.2f} s\nstrict {result['strict']}; contact {result['contact']}; late motion {result['late_motion']}"
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=130)
    for label, center in [
        ("hold_entry", 0),
        *([("first_failure", min(crossings))] if crossings else []),
    ]:
        for axis in axes:
            axis.set_xlim(max(min(time), center - 1), min(max(time), center + 1))
        figure.savefig(destination.with_stem(destination.stem + "_" + label), dpi=130)
    plt.close(figure)


def paired_material_comparisons(state, results):
    return [
        row
        for stage in ("screen", "60s", "120s")
        for row in stage_material_comparisons(results, stage)
    ]


def stage_material_comparisons(results, stage):
    screens = {r["condition"]: r for r in results if r["stage"] == stage}
    comparisons = []
    for geometry in ("G0", "G6"):
        for mu in (0.6, 0.8, 1.2):
            names = [
                f"{geometry}_{scope}_{mu}".replace(".", "p") for scope in ("H", "M")
            ]
            pair = [screens.get(name) for name in names]
            comparisons.append(
                dict(
                    geometry=geometry,
                    mu=mu,
                    stage=stage,
                    conditions=names,
                    **compare_pair(pair),
                )
            )
    return comparisons


def compare_pair(pair):
    if not all(p and p["matched"]["status"] == "MEASURED" for p in pair):
        return {"status": "NOT RUN"}
    return {
        "status": "MEASURED",
        "H": pair[0]["matched"],
        "M": pair[1]["matched"],
        "H_foot_slip": pair[0]["matched_foot_slip"],
        "M_foot_slip": pair[1]["matched_foot_slip"],
        "hand_load_change_over_50_percent": {
            side: abs(
                pair[1]["matched"]["mean_hand_load_fraction"][side]
                - pair[0]["matched"]["mean_hand_load_fraction"][side]
            )
            > 0.5 * abs(pair[0]["matched"]["mean_hand_load_fraction"][side])
            for side in ("left", "right")
        },
    }


def report(root, output):
    state = read(root / "campaign_state.json")
    output.mkdir(parents=True, exist_ok=True)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    results = []
    for record in state["runs"]:
        if record["status"] == "COMPLETE":
            results.append(analyze_record(root, record, output, plots))
    statuses = {
        name: condition_status(
            [
                r
                for r in state["runs"]
                if r["candidate"] == name and r["status"] == "COMPLETE"
            ]
        )
        for name in state["candidates"]
    }
    data = {
        "conditions": statuses,
        "runs": results,
        "H_vs_M": paired_material_comparisons(state, results),
        "decisions": state["decisions"],
    }
    (output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    write_table(state, results, statuses, output)
    write_matrix(results, output)
    write_timeline(results, output)
    return data


def analyze_record(root, record, output, plots):
    cache = output / "cache" / (record["id"] + ".json")
    identity = {
        "analysis_program_sha256": file_hash(Path(__file__)),
        "run_files_sha256": record["files_sha256"],
    }
    existing = cached_result(cache, identity, plots)
    if existing is not None:
        return existing
    result, rows = summarize(root, record)
    plot_run(rows, result, plots / (record["id"] + ".png"))
    cache.parent.mkdir(exist_ok=True)
    value = {
        "identity": identity,
        "result": result,
        "plot_sha256": {
            path.name: file_hash(path) for path in plots.glob(record["id"] + "*.png")
        },
    }
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(json.dumps(value) + "\n")
    temporary.replace(cache)
    return result


def cached_result(path, identity, plots):
    if not path.exists():
        return None
    value = read(path)
    if value["identity"] != identity:
        return None
    for name, expected in value["plot_sha256"].items():
        plot = plots / name
        if not plot.exists() or file_hash(plot) != expected:
            return None
    return value["result"]


def write_matrix(results, output):
    screens = {r["condition"]: r for r in results if r["stage"] == "screen"}
    columns = [
        "reference",
        "H_0p4",
        "H_0p6",
        "H_0p8",
        "H_1p2",
        "M_0p6",
        "M_0p8",
        "M_1p2",
    ]
    lines = [
        "# Geometry by friction screens",
        "",
        "Each measured cell is strict/contact status and achieved HOLD seconds. NR = NOT RUN. Native reference uses mu=1.8; H at 1.8 is a verified alias for G0/G5/G6.",
        "",
        "| Geometry | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for geometry in range(7):
        cells = []
        for column in columns:
            run = screens.get(f"G{geometry}_{column}")
            cells.append(
                f"{run['strict']}/{run['contact']} {run['achieved_s']:.2f}s"
                if run
                else "NR"
            )
        lines.append(f"| G{geometry} | " + " | ".join(cells) + " |")
    (output / "MATRIX.md").write_text("\n".join(lines) + "\n")


def write_timeline(results, output):
    fields = (
        "condition",
        "stage",
        "event_scope",
        "event",
        "phase",
        "first_absolute_s",
        "first_hold_s",
        "persistent_absolute_s",
        "persistent_hold_s",
        "terminal_cause",
    )
    with (output / "failure_timeline.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            entry = hold_entry(result["summary"])
            for scope, crossing in timeline_events(result):
                event = crossing["event"]
                onset = crossing.get("persistence_onset_s")
                first = crossing["first_crossing_s"]
                writer.writerow(
                    {
                        "condition": result["condition"],
                        "stage": result["stage"],
                        "event_scope": scope,
                        "event": event,
                        "phase": crossing["phase"],
                        "first_absolute_s": first,
                        "first_hold_s": first - entry if entry is not None else None,
                        "persistent_absolute_s": onset,
                        "persistent_hold_s": onset - entry
                        if onset is not None and entry is not None
                        else None,
                        "terminal_cause": ", ".join(result["terminal_cause"]),
                    }
                )


def timeline_events(result):
    events = [("all phases; diagnostic", event) for event in result["all_phase_events"]]
    for name, crossing in result["first_crossings"].items():
        onset = result["first_persistent"].get(name, {}).get("persistence_onset_s")
        events.append(("scored HOLD", {**crossing, "persistence_onset_s": onset}))
    return events


def write_table(state, results, statuses, output):
    fields = (
        "condition",
        "stage",
        "requested_s",
        "achieved_s",
        "strict",
        "contact",
        "late_motion",
        "maximum_reset_drift_m",
        "terminal_cause",
    )
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in results)
    lines = [
        "# Ladder/material measured results",
        "",
        "Software checks and physical qualification are separate. Every promoted run starts from a fresh reset.",
        "",
        "| Condition | Stage | HOLD s | Strict | Contact | Late motion | Terminal cause |",
        "|---|---|---:|---|---|---|---|",
    ]
    lines.extend(
        f"| {r['condition']} | {r['stage']} | {r['achieved_s']:.2f} | {r['strict']} | {r['contact']} | {r['late_motion']} | {', '.join(r['terminal_cause'])} |"
        for r in results
    )
    lines += [
        "",
        "## Qualification by tested condition",
        "",
        "| Condition | Status |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {status} |" for name, status in statuses.items())
    lines += [
        "",
        "H at mu=1.8 aliases each geometry's native reference only where compiled equivalence was verified. Other unlisted geometry/friction combinations are NOT RUN. Incomplete runs have no matched-window average. A small hand-load fraction is not evidence of hand capacity.",
    ]
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/g1_o6/ladder_material")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/g1_o6/ladder_material/results")
    )
    args = parser.parse_args()
    report(args.root, args.output)
