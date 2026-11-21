"""Publish matched G6 traces, physical gate times and source hashes from the completed campaigns."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.g1_o6_g6_thumb_replay import describe, json_values, loads, replay
from teleopit.sim.g1_o6_support.ladder_campaign import file_hash
from teleopit.sim.g1_o6_support.g6_wrist_grip_experiment import passes


def first_failure(*clocks):
    events = [(round(t, 6), k) for clock in clocks for k, t in clock.items() if t is not None]
    if not events:
        return None
    first = min(t for t, _ in events)
    return {"hold_s": first, "conditions": sorted({k for t, k in events if t == first})}


def control_failures(path):
    first = {}
    with (path / "steps.jsonl").open("rb") as stream:
        for row in map(loads, stream):
            if row["phase"] == "HOLD":
                for reason in row["failure_reasons"]:
                    first.setdefault(reason, row["hold_time_s"])
    return first


def audited_failures(result):
    audit = result["physics_audit"]
    start = next(e["time"] for e in result["strict"]["events"] if e["next"] == "HOLD")
    clock = {e["event"]:e["first_crossing_s"] - start for e in audit["first_crossings"].values()
             if e["phase"] == "HOLD"}
    # Anatomical thumb timings are independently measured from native records.
    for side in ("left", "right"):
        key = "opposition_loss:" + side + "_hand"
        if key in clock:
            clock[side + "_thumb"] = clock.pop(key)
    for name, when in audit["grip_support"]["control"]["first_loss_s"].items():
        if when is not None:
            clock[name] = min(clock.get(name, when), when)
    return clock


RAW = ("scene.mjb", "steps.jsonl", "physics_contacts.jsonl.gz", "knee_rung3.f64.gz", "resolved.json", "summary.json")


def cached_signals(path, record, cache):
    hashes = record["files_sha256"]
    for name in RAW:
        if file_hash(path / name) != hashes[name]:
            raise ValueError(f"changed raw evidence: {path / name}")
    key = hashlib.sha256("".join(hashes[k] for k in RAW).encode()).hexdigest()
    file = cache / (key + ".npz")
    if not file.exists():
        arrays = replay(path)
        temp = file.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **arrays)
        temp.replace(file)
    with np.load(file) as saved:
        return {k:saved[k] for k in saved}


def trial_evidence(path, record, data):
    result = record["result"]
    audit = result["physics_audit"]
    evidence = describe(data)
    if len(data["time"]) != audit["scored_steps"]:
        raise ValueError("incomplete native hold trace")
    original = audited_failures(result)
    first = first_failure(evidence["first"], original, control_failures(path))
    stage = record["stage"]
    passed = passes(result, "60s" if stage == "repeat60" else stage)
    if not passed and not first:
        first = {"hold_s":evidence["end_s"], "conditions":["stage qualification at completion"]}
    return {**evidence, "run_id":record["id"], "candidate":record["candidate"], "stage":stage,
            "qualified_stage":bool(passed), "first_failure":first, "audited_threshold_crossings_s":original,
            "peak_residual_N":float(np.max(data["residual"])), "minimum_knee_mm":float(1000*np.min(data["knee_gap"])),
            "nonzero_applied_steps":audit["nonzero_applied_steps"], "housing_steps":audit["housing_contact_steps"],
            "unintended_steps":audit["unintended_support_steps"], "native_friction":audit["native_hand_friction"]["status"],
            "identity":record["identity"], "files_sha256":record["files_sha256"]}


def compare_plot(pairs, output):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    colors = ("#14577b", "#ae431d")
    for (name, a), color in zip(pairs, colors, strict=True):
        t, ct = a["time"], a["control_time"]
        for key, style in (("centroid_margin", "-"), ("fixed_margin", "--")):
            axes[0].plot(t, 1000 * a[key], style, color=color, lw=1.45)
        for key, style in (("requested", "-"), ("applied", "--")):
            axes[1].plot(ct, 1000 * np.linalg.norm(a[key], axis=1), style, color=color, lw=1.25)
        palm = a["right_palm"] - a["right_palm"][np.argmin(np.abs(t-5))]
        for axis, style in enumerate(("-", ":", "--")):
            axes[2].plot(t, 1000*palm[:, axis], style, color=color, lw=1.45)
        axes[0].plot([], [], color=color, lw=3, label=name)
        lost = np.flatnonzero((a["right_thumb"] < .05) | (a["left_thumb"] < .05))
        if lost.size:
            event = t[lost[0]]
            for ax in axes:
                ax.axvline(event, color=color, alpha=.55, lw=1, linestyle=":")
            axes[0].annotate(f"{event:.3f} s", (event, .05), xycoords=("data","axes fraction"),
                             xytext=(-4,2), textcoords="offset points", ha="right", color=color, fontsize=9)
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, 1.54), ncol=2, frameon=False)
    axes[0].set(ylabel="Edge clearance (mm)", title="Right thumb   ·   solid contact centroid   /   dashed fixed pad point")
    axes[1].axhline(30, color="#454545", alpha=.55, lw=1, ls=":")
    axes[1].set(ylabel="Rotation (mrad)", title="Right wrist   ·   solid requested before Cartesian limits   /   dashed applied")
    axes[2].set(ylabel="Displacement (mm)", xlabel="Time after assistance removal (s)",
                title="Right palm relative to tread at 5 s   ·   solid x   /   dotted y   /   dashed z")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
        ax.margins(x=.01)
        ax.title.set_fontsize(10)
    fig.suptitle("G6 thumb retention during unsupported hold", y=.98, fontsize=15)
    fig.subplots_adjust(top=.84, bottom=.09, left=.12, right=.98, hspace=.44)
    fig.savefig(output, dpi=175, facecolor="white")
    plt.close(fig)


def read_campaign(root, cache):
    state = loads((root / "campaign_state.json").read_bytes())
    trials, traces = [], {}
    for record in state["runs"]:
        if record["status"] != "COMPLETE":
            raise ValueError("campaign has unfinished runs")
        path = root / record["output"]
        data = cached_signals(path, record, cache)
        trials.append(trial_evidence(path, record, data))
        traces[record["id"]] = data
    return trials, traces


def publish(campaign, reference, cache, output):
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    experiments, candidates = read_campaign(campaign, cache)
    repetitions, references = read_campaign(reference, cache)
    longest = max((r for r in experiments if r["candidate"] != "control"), key=lambda r:r["end_s"])
    compare_plot([("Right-wrist reference", references["control_60s"]),
                  ("Right-arm " + longest["candidate"], candidates[longest["run_id"]])], output / "comparison.png")
    metrics = {"experiments":experiments, "reference_repeats":repetitions,
               "milestones":loads((campaign / "milestones.json").read_bytes())}
    (output / "results.json").write_text(json.dumps(json_values(metrics), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "reference", "cache", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    publish(args.campaign, args.reference, args.cache, args.output)
