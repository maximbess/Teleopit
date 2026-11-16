"""Candidate tables and full-acquisition plots from immutable G6 traces."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def summarize(record, candidate):
    result = record.get("result", {})
    strict = result.get("strict", {})
    audit = result.get("physics_audit", {})
    return {
        "candidate": record["candidate"],
        "stage": record["stage"],
        "run": record["output"],
        "pose": candidate["pose"],
        "reset": candidate["reset"],
        "status": record["status"],
        "reset_status": result.get("reset_status", "FEASIBLE"),
        "achieved_hold_s": strict.get("unsupported_hold_seconds", 0),
        "strict": result.get("strict_status", "ERROR"),
        "contact": result.get("contact_quality", {}).get("status", "NOT RUN"),
        "failure": strict.get("failure_reasons", result.get("reasons", [])),
        "final_reset_drift_m": strict.get("final_diagnostics", {}).get("base_drift_m"),
        "short_drift": audit.get("short_drift"),
        "minimum_clearance_m": audit.get("body_minimum_clearance_m"),
        "peak_effort_fraction": strict.get("peak_effort_fraction"),
        "next_tread_reach": audit.get("next_tread_reach"),
        "first_crossings": audit.get("first_crossings", {}),
        "contact_losses": result.get("contact_quality", {}).get(
            "persistent_quality_losses_s", {}
        ),
    }


def trace(path):
    rows = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            weight = row["support_measurement"]["weight_N"]
            hands = row.get("pose_search", {}).get("hands", {})
            rows.append(
                {
                    "time": row["time"],
                    "phase": row["phase"],
                    "drift": row["base_drift_m"],
                    "assist": row["assistance_gain"],
                    "supports": {
                        s: v["force_world"][2] for s, v in row["supports"].items()
                    },
                    "shares": {
                        s: row["supports"][s + "_hand"]["force_world"][2] / weight
                        for s in ("left", "right")
                    },
                    "opposition": {
                        s: hands.get(s, {}).get("quality", {}).get("valid", False)
                        for s in ("left", "right")
                    },
                    "clearance": {
                        name: min(v["signed_distance_m"] for v in obstacles.values())
                        for name, obstacles in row.get("g6", {})
                        .get("body_clearance", {})
                        .items()
                    },
                    "effort": max(row["effort_fraction"]),
                    "residual": float(np.linalg.norm(row["static_balance_residual_N"])),
                }
            )
    return rows


def plot(rows, label, output):
    time = [r["time"] for r in rows]
    figure, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    flat = axes.ravel()
    flat[0].axhline(30, color="red", linestyle="--")
    for axis, spec in zip(flat, chart_specs(rows)):
        draw_chart(axis, time, rows, *spec)
    figure.suptitle(label)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def chart_specs(rows):
    hands = ("left", "right")
    return [
        ("Base displacement (mm)", [("drift",)], 1000, False),
        ("Measured hand weight share (%)", [("shares", s) for s in hands], 100, False),
        ("Anatomical validity", [("opposition", s) for s in hands], 1, True),
        ("Nearest tread/rail clearance (mm)", [("clearance", s + "_" + part)
            for s in hands for part in ("knee", "shin")], 1000, False),
        ("Actuator effort / physical limit", [("effort",)], 1, False),
        ("Acquisition assistance", [("assist",)], 1, False),
        ("Signed external vertical support force (N)", [("supports", s) for s in rows[0]["supports"]], 1, False),
        ("Static force balance residual (N)", [("residual",)], 1, False)]


def draw_chart(axis, time, rows, title, fields, scale, stepped):
    draw = axis.step if stepped else axis.plot
    for path in fields:
        values = np.array([nested_value(row, path) for row in rows], dtype=float)
        draw(time, scale * values, label=path[-1])
    axis.set_title(title)
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    axis.set_xlabel("Simulator time (s), including acquisition")


def nested_value(row, path):
    for key in path:
        row = row.get(key, np.nan)
    return row


def candidate_table(records, output):
    headers = (
        "candidate",
        "stage",
        "reset_status",
        "achieved_hold_s",
        "strict",
        "contact",
        "failure",
        "final_reset_drift_m",
        "peak_effort_fraction",
        "minimum_clearance_m",
        "pose",
        "reset",
    )
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(record[key])
                    if isinstance(record[key], (dict, list))
                    else record[key]
                    for key in headers
                }
            )


def report(root):
    state = read(root / "campaign_state.json")
    output = root / "report"
    output.mkdir(exist_ok=True)
    records = [
        summarize(r, state["candidates"][r["candidate"]]) for r in state["runs"]
    ]
    (output / "candidates.json").write_text(json.dumps(records, indent=2))
    candidate_table(records, output / "candidates.csv")
    for record in records:
        path = root / record["run"] / "steps.jsonl"
        if record["status"] == "COMPLETE" and path.exists():
            rows = trace(path)
            plot(
                rows,
                f"{record['candidate']} {record['stage']} | strict {record['strict']} | contact {record['contact']}",
                output / f"{record['candidate']}_{record['stage']}.png",
            )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/g1_o6/g6_stability"))
    report(parser.parse_args().root)
