"""Publish four-support comparisons and cycle outcomes from native rung-7 recordings."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.g1_o6_g6_reference_replay import decoded_hash
from teleopit.sim.g1_o6_support.ladder_campaign import file_hash
from teleopit.sim.g1_o6_support.regrasp import write_json
from teleopit.sim.g1_o6_support.regrasp_measure import STATES, SUPPORTS
from teleopit.sim.g1_o6_support.regrasp_results import (
    contact_events,
    failures,
    interval,
    read_run,
    score_pair,
)


def csv_file(path, rows):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cycle(outcome, cfg, t):
    events = contact_events(t, outcome)
    start = (
        events.get("clear_verified1")
        if cfg["pelvis"] == "fixed"
        else events.get("settled")
    )
    if start is None:
        return {
            "moving_completed": False,
            "supported_cycle": False,
            "events": events,
            "missing": outcome["missing"],
        }
    active = t["time"] >= start - 1e-8
    first = failures(t, cfg, start)
    i = int(cfg["side"] == "right")
    moving = (
        all(key in outcome["windows"] for key in ("hold1", "hold2"))
        and "clear_verified2" in events
    )
    opp = 1 - i
    drops = (t["state"][:, opp] != STATES.index("grasp")) & active
    verified_gaps, travel = {}, {}
    for n in ("1", "2"):
        if "clear_verified" + n in events:
            stop = events["clear_verified" + n]
            mask = interval(t["time"], (stop - 0.5, stop))
            verified_gaps[n] = float(min(t["gap"][mask]))
            base = np.argmin(abs(t["time"] - events["open_request" + n]))
            travel[n] = float(
                t["wrist"][base, i, 0] - t["wrist"][np.flatnonzero(mask)[-1], i, 0]
            )
    return {
        "moving_completed": moving,
        "supported_cycle": moving and not any(v is not None for v in first.values()),
        "first": first,
        "events": events,
        "missing": outcome["missing"],
        "handoffs": outcome["handoffs"],
        "minimum_verified_gap_m": verified_gaps,
        "measured_retreat_m": travel,
        "max_pelvis_drift_m": t["drift"][active].max(),
        "max_tilt_rad": t["tilt"][active].max(),
        "max_applied": t["applied"][active].max(),
        "max_fixture_force_N": np.linalg.norm(t["fixture"][active, :3], axis=1).max(),
        "mean_fixture_Fz_N": t["fixture"][active, 2].mean(),
        "opposite_grasp_loss_s": float(drops.sum() * cfg["config"]["physics_dt"]),
        "min_foot_Fz_N": t["fz"][active, 2:].min(axis=0).tolist(),
    }


def collect(root):
    results, pairs, plots, rows = {}, {}, defaultdict(list), []
    for path in sorted(root.glob("*/release/outcome.json")):
        directory = path.parent.parent
        outcome, cfg, trace = read_run(path.parent)
        record = cycle(outcome, cfg, trace)
        results[directory.name] = dict(
            pelvis=cfg["pelvis"], side=cfg["side"], offset_mm=cfg["offset_mm"], **record
        )
        if cfg["pelvis"] == "free" and (directory / "control/outcome.json").is_file():
            pair = score_pair(directory)
            pairs[directory.name] = {k: v for k, v in pair.items() if k != "forces"}
            rows.extend(
                dict(trial=directory.name, side=cfg["side"], **f)
                for f in pair["forces"]
            )
            _, _, control = read_run(directory / "control")
            baseline = outcome["windows"]["baseline"]
            mask = trace["time"] >= baseline[0] - 1e-8
            plots[cfg["side"]].append(
                (
                    trace["time"][mask] - baseline[1],
                    trace["fz"][mask],
                    control["fz"][mask],
                    pair["events"],
                    pair["release_first_events"],
                    baseline[1],
                )
            )
    return results, pairs, plots, rows


def summary(records):
    groups = defaultdict(list)
    for record in records.values():
        groups[record["pelvis"], record["side"]].append(record)
    return [
        cycle_counts(mode, side, groups[mode, side])
        for mode in ("fixed", "free")
        for side in ("left", "right")
        if (mode, side) in groups
    ]


def cycle_counts(mode, side, selected):
    nominal = [r for r in selected if r["offset_mm"] == 0]
    offsets = sorted({r["offset_mm"] for r in selected if r["moving_completed"]})
    return {
        "pelvis": mode,
        "hand": side,
        "nominal_repeats": len(nominal),
        "moving_cycles": sum(r["moving_completed"] for r in nominal),
        "continuously_supported_cycles": sum(r["supported_cycle"] for r in nominal),
        "passed_moving_offsets_mm": " ".join(map(str, offsets)),
        "first_missing": next((r["missing"] for r in nominal if r["missing"]), ""),
    }


def force_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["side"], row["window"], row["support"]].append(row)
    result = []
    for (side, window, support), repeats in groups.items():
        effects = [r["effect_N"] for r in repeats]
        result.append(
            {
                "side": side,
                "window": window,
                "support": support,
                "n": len(repeats),
                "status": repeats[0]["status"],
                "mean_effect_N": np.mean(effects),
                "mean_effect_pp": np.mean([r["effect_pp"] for r in repeats]),
                "between_repeat_range_N": np.ptp(effects),
                "raw_min_N": min(r["min_N"] for r in repeats),
                "raw_max_N": max(r["max_N"] for r in repeats),
            }
        )
    return result


def event_marks(events, violations, origin):
    marks = [
        (events.get(k), label)
        for k, label in (
            ("open_request1", "Open"),
            ("opened1", "Opened"),
            ("last_contact_end1", "Last contact"),
            ("clear_verified1", "Clear"),
            ("approach1", "Approach"),
            ("grasp_verified1", "Grasp"),
            ("recovered1", "Recovery"),
            ("open_request2", "Open 2"),
            ("opened2", "Opened 2"),
            ("last_contact_end2", "Last contact 2"),
            ("clear_verified2", "Clear / return 2"),
            ("grasp_verified2", "Regrasp 2"),
            ("recovered2", "Recovery 2"),
        )
    ]
    losses = [v for k, v in violations.items() if k.endswith("_loss") and v is not None]
    if losses:
        marks.append((min(losses), "Support loss"))
    return sorted((t - origin, label) for t, label in marks if t is not None)


def event_caption(marks):
    first, second = [], []
    for when, label in marks:
        line = second if label.endswith(" 2") else first
        line.append(f"{when:.3f} s  {label.lower()}")
    return "\n".join("    ".join(line) for line in (first, second) if line)


def force_line(ax, t, values, color, label):
    # Keep every transient peak visible. The darker line averages 101 native samples.
    ax.fill_between(
        t, np.min(values, axis=0), np.max(values, axis=0), color=color, alpha=0.2
    )
    for trace in values:
        ax.plot(t, trace, color=color, lw=0.5, alpha=0.27)
        trend = np.convolve(trace, np.ones(101) / 101, mode="valid")
        ax.plot(t[50:-50], trend, color=color, lw=1.0, label=label)
        label = None


def plot_support(ax, diff, runs, i, arrays):
    name = SUPPORTS[i]
    differences = []
    for j, (t, r, c, *_) in enumerate(runs):
        arrays[f"release_{j + 1}_{name}_N"], arrays[f"control_{j + 1}_{name}_N"] = (
            r[:, i],
            c[:, i],
        )
        arrays[f"time_{j + 1}_s"] = t
        baseline = (t >= -2.0 - 1e-8) & (t < -1e-8)
        net = r[:, i] - c[:, i]
        differences.append(net - net[baseline].mean())
    # The repeats may end at different timeouts. Bands use their common prefix.
    n = min(map(len, differences))
    time = runs[0][0][:n]
    force_line(ax, time, [run[2][:n, i] for run in runs], "#63656b", "Both hold")
    force_line(ax, time, [run[1][:n, i] for run in runs], "#087b8d", "Release")
    force_line(diff, time, [v[:n] for v in differences], "#884678", None)
    diff.axhline(0, color="0.7", lw=0.6)
    ax.set_ylabel(name.replace("_", " ").title())


def figure(output, side, runs):
    time, _release, _control, events, violations, origin = runs[0]
    fig, axes = plt.subplots(4, 2, figsize=(12, 8), sharex=True, layout="constrained")
    fig.suptitle(
        f"{side.title()} hand release on rung 7   ·   free pelvis", fontsize=17
    )
    marks = event_marks(events, violations, origin)
    arrays = {"time_s": time}
    axes[0, 0].set_title("Vertical reaction (N)", loc="left")
    axes[0, 1].set_title("Change relative to matched hold (N)", loc="left")
    for i in range(len(SUPPORTS)):
        ax, diff = axes[i]
        plot_support(ax, diff, runs, i, arrays)
        for cell in (ax, diff):
            for n in ("1", "2"):
                if "recovered" + n in events:
                    cell.axvspan(
                        events["grasp_verified" + n] - origin,
                        events["recovered" + n] - origin,
                        color="#ebc984",
                        alpha=0.25,
                        lw=0,
                    )
            if "clear_verified1" in events and "approach1" in events:
                cell.axvspan(
                    events["clear_verified1"] - origin,
                    events["approach1"] - origin,
                    color="#c7e6dd",
                    alpha=0.22,
                    lw=0,
                )
            for when, _ in marks:
                cell.axvline(when, color=".55", lw=0.65, ls=":")
            cell.spines[["top", "right"]].set_visible(False)
            cell.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(loc="best", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Seconds from open request")
    fig.supxlabel(event_caption(marks), fontsize=8)
    fig.savefig(output / f"{side}_forces.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(output / f"{side}_forces.npz", **arrays)


def publish(root, output):
    output.mkdir(parents=True, exist_ok=True)
    records, pairs, plots, rows = collect(root)
    write_json(output / "outcomes.json", records)
    write_json(output / "matched_events.json", pairs)
    csv_file(output / "cycles.csv", summary(records))
    if rows:
        csv_file(output / "paired_forces.csv", rows)
        csv_file(output / "paired_repeats.csv", force_summary(rows))
    for side, runs in plots.items():
        figure(output, side, runs)
    evidence_ledger(root, output, records)


def evidence_ledger(root, output, records):
    runs = [
        d
        for name in records
        for d in (root / name).iterdir()
        if (d / "outcome.json").is_file()
    ]
    runs.extend(sorted(root.glob("neg_*/")))
    ledger = {
        str(d.relative_to(root).as_posix()): {
            p.name: file_hash(p) for p in sorted(d.iterdir()) if p.is_file()
        }
        for d in runs
    }
    write_json(output / "raw_sha256.json", ledger)
    reference = root / "reference"
    if (reference / "manifest.json").is_file() and runs:
        source = Path(json.loads((runs[0] / "resolved.json").read_bytes())["source"])
        write_json(output / "reference.json", reference_receipt(reference, source))


def reference_receipt(reference, source):
    meta = json.loads((reference / "resolved.json").read_bytes())
    native = {}
    for name in (
        "physics_contacts.jsonl.gz",
        "knee_rung3.f64.gz",
        "applied_forces.jsonl.gz",
    ):
        native[name] = decoded_hash(reference / name)
        if native[name] != decoded_hash(source / name):
            raise ValueError("reference solver stream changed: " + name)
    steps = file_hash(reference / "steps.jsonl")
    if steps != file_hash(source / "steps.jsonl"):
        raise ValueError("reference command/state trace changed")
    with (reference / "steps.jsonl").open() as stream:
        forces = [
            [row["supports"][k]["force_world"][2] for k in SUPPORTS]
            for row in map(json.loads, stream)
            if row["phase"] == "HOLD"
        ]
    return {
        "physics": meta["physical_identity"],
        "source_model_sha256": file_hash(source / "scene.mjb"),
        "controller": meta["environment_spec"]["grasp_acquisition"],
        "seed": meta["seed"],
        "control_dt": meta["config"]["control_dt"],
        "physics_dt": meta["config"]["physics_dt"],
        "hold_endpoints": len(forces),
        "mean_Fz_N": dict(zip(SUPPORTS, np.mean(forces, axis=0), strict=True)),
        "steps_sha256": steps,
        "decoded_native_sha256": native,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    publish(args.root, args.output)
