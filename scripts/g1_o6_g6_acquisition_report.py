"""Independently check acquisition anatomy from native contact streams."""

import argparse
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.g1_o6_g6_report import read, report
from scripts.g1_o6_g6_arm_report import verify_run as verify_arm
from teleopit.sim.g1_o6_support.pose_contacts import FINGERS, contact_group


def selected_contacts(row, side):
    target = row["support_targets"][side + "_hand"]
    region = side + "_hand"
    result = []
    for contact in row["contacts"]:
        if target not in contact["geoms"] or region not in contact["regions"] or contact["normal_force"] < .05:
            continue
        index = contact["regions"].index(region)
        result.append({"group": contact_group(contact["bodies"][index]),
            "normal": np.array(contact["normal_on_second_world"]) * (1 if index == 1 else -1),
            "z": contact["position"][2], "force": contact["normal_force"]})
    return result


def ordered_contacts(groups, height):
    return all(any((c["z"] - height) * (-1 if group == "thumb" else 1) > 0 for c in groups[group]) for group in FINGERS)


def opposed_contacts(groups):
    thumb = [c["normal"] for c in groups["thumb"]]
    fingers = [c["normal"] for group in FINGERS[1:] for c in groups[group]]
    return bool(thumb and fingers and np.min(np.array(thumb) @ np.array(fingers).T) < -.5)


def raw_hand(row, side, height):
    contacts = selected_contacts(row, side)
    groups = {group: [c for c in contacts if c["group"] == group] for group in FINGERS}
    ordered, opposed = ordered_contacts(groups, height), opposed_contacts(groups)
    return {"valid": ordered and opposed,
            "thumb_normal_N": sum(c["force"] for c in groups["thumb"]),
            "minimum_finger_normal_N": min(sum(c["force"] for c in groups[g]) for g in FINGERS[1:])}


def anatomy_sample(phases, row, heights):
    hands = {side: raw_hand(row, side, height) for side, height in heights.items()}
    values = phases.setdefault(row["phase"], {"samples": 0, "anatomical_samples": 0,
        "tail_samples": 0, "peak_thumb_normal_N": {side: 0. for side in heights}})
    valid = all(h["valid"] for h in hands.values())
    values["samples"] += 1
    values["anatomical_samples"] += int(valid)
    values["tail_samples"] = values["tail_samples"] + 1 if valid else 0
    for side in heights:
        values["peak_thumb_normal_N"][side] = max(values["peak_thumb_normal_N"][side], hands[side]["thumb_normal_N"])


def check_phase(values, expected):
    assert values["samples"] == expected["samples"]
    assert values["anatomical_samples"] >= expected["valid_samples"]
    assert values["tail_samples"] >= expected["tail_samples"]
    np.testing.assert_allclose(list(values["peak_thumb_normal_N"].values()), list(expected["peak_thumb_normal_N"].values()), atol=1e-9)


def check_report(actual, reported, dt):
    assert set(actual) == set(reported["phases"])
    for phase, values in actual.items():
        check_phase(values, reported["phases"][phase])
    if reported["ready_for_removal"]:
        assert actual["ACQUIRE"]["tail_samples"] * dt >= .5 - 1e-9
    if reported["ready_for_hold"]:
        assert actual["REMOVE_ASSISTANCE"]["tail_samples"] * dt >= .5 - 1e-9


def verify_anatomy(path, metadata, model):
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    heights = {side: data.geom(metadata["support_targets"][side + "_hand"]).xpos[2] for side in ("left", "right")}
    phases = {}
    with gzip.open(path / "physics_contacts.jsonl.gz", "rt") as stream:
        for line in stream:
            anatomy_sample(phases, json.loads(line), heights)
    reported = read(path / "summary.json")["physics_audit"]["grasp_acquisition"]
    check_report(phases, reported, metadata["config"]["physics_dt"])
    return {"status": "PASS", "phases": phases,
            "claimed_ready_for_hold": reported["ready_for_hold"],
            "method": "native per-physics-step contacts, >=0.05 N, thumb/finger normal opposition and world-height ordering; no controller-state claims"}


def verify_targets(path, metadata, model):
    limits = model.jnt_range[model.actuator_trnid[:, 0]]
    fingers = np.array([model.actuator(name + "_servo").id for hand in metadata["model"]["hands"].values() for name in hand["active"]])
    body = np.setdiff1d(np.arange(model.nu), fingers)
    profile = metadata["environment_spec"].get("grasp_acquisition", {})
    peak_rate, count = 0., 0
    with (path / "steps.jsonl").open() as stream:
        initial = json.loads(next(stream))
        baseline = np.asarray(initial["applied_position_targets_rad"])
        previous, time = baseline, initial["time"]
        for line in stream:
            row = json.loads(line)
            command = np.asarray(row["applied_position_targets_rad"])
            assert np.all(np.isfinite(command))
            assert np.all(command >= limits[:, 0] - 1e-9) and np.all(command <= limits[:, 1] + 1e-9)
            rate = float(np.max(abs(command - previous)) / (row["time"] - time))
            assert rate <= 1. + 1e-8
            if not profile.get("arm_rate_rad_s", 0):
                np.testing.assert_array_equal(command[body], baseline[body])
            peak_rate = max(peak_rate, rate)
            previous, time = command, row["time"]
            count += 1
    assert count > 0
    return {"status": "PASS", "samples": count, "maximum_target_rate_rad_s": peak_rate}


def verify_run(path, output):
    if "physics_audit" not in read(path / "summary.json"):
        return {"status": "NOT_RUN", "reason": "reset rejected; no dynamic evidence"}
    metadata = read(path / "resolved.json")
    model = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
    return {"anatomy": verify_anatomy(path, metadata, model),
            "commands": verify_targets(path, metadata, model),
            "arm_bounds": verify_arm(path, output)}


def plot_contacts(path, output):
    rows = []
    with (path / "steps.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            rows.append((row["time"], {side: hand["quality"] for side, hand in row["pose_search"]["hands"].items()}))
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for column, side in enumerate(("left", "right")):
        for axis, key in zip(axes[:, column], ("thumb_normal_N", "minimum_finger_normal_N")):
            axis.plot([r[0] for r in rows], [r[1][side][key] for r in rows])
            axis.axhline(.05, color="red", linestyle="--")
            axis.axvspan(3.04, 5.04, color="grey", alpha=.15)
            axis.set_title(side + " " + key)
            axis.set_xlabel("Simulator time (s), including acquisition")
            axis.grid(alpha=.2)
    figure.suptitle(path.name + " | measured intended-tread normal forces")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def verify(root):
    report(root)
    state = read(root / "campaign_state.json")
    checks = {}
    for record in state["runs"]:
        path = root / record["output"]
        output = root / "report" / (record["id"] + "_commands.json")
        checks[record["id"]] = verify_run(path, output)
        if (path / "steps.jsonl").exists():
            plot_contacts(path, root / "report" / (record["id"] + "_acquisition.png"))
    (root / "report/acquisition_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "runs": len(checks)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/g1_o6/g6_grasp_acquisition"))
    verify(parser.parse_args().root)
