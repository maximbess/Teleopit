"""Stream preserved traces into comparable, side-specific screening evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import DriftCriteria


def read(path):
    return json.loads(Path(path).read_text())


def hold_entry(summary):
    return next(
        (
            e["time"]
            for e in summary.get("strict", {}).get("events", [])
            if e["next"] == "HOLD"
        ),
        None,
    )


def hand_signals(row, side):
    hand = row["pose_search"]["hands"][side]
    motion, quality = hand["motion"], hand["quality"]
    yaw = next(
        i
        for i, name in enumerate(motion["joint_names"])
        if name.endswith("_thumb_cmc_yaw")
    )
    wrist = np.array(row["wrist_world_on_hand"][side])
    pose = row["diagnostic_v1"]["rung_relative_poses_v1"][side]
    return {
        "thumb_yaw_rad": motion["position_rad"][yaw],
        "thumb_target_rad": motion["applied_target_rad"][yaw],
        "thumb_raw_target_rad": motion["raw_target_rad"][yaw],
        "thumb_travel_rad": motion["hold_travel_rad"][yaw],
        "thumb_work_J": motion["hold_work_J"][yaw],
        "thumb_power_W": motion["actuator_power_W"][yaw],
        "thumb_normal_N": quality["thumb_normal_N"],
        "finger_min_normal_N": quality["minimum_finger_normal_N"],
        "thumb_centroid_m": quality["centroid_by_group_m"]["thumb"],
        "valid": quality["valid"],
        "palm_position_m": pose["palm"]["position_m"],
        "mount_position_m": pose["mount"]["position_m"],
        "wrist_force_N": float(np.linalg.norm(wrist[:3])),
        "wrist_moment_Nm": float(np.linalg.norm(wrist[3:])),
        "share": row["support_measurement"]["per_hand_share"][side],
        "joint_margin_rad": min(motion["joint_stop_margin_rad"]),
        "effort_margin_Nm": min(motion["effort_margin_Nm"]),
        "wrist_bend_deg": row["posture"]["wrist_bend_deg"][side],
        "elbow_bend_deg": row["posture"]["elbow_bend_deg"][side],
        "housing_clearance_m": row["posture"]["housing_clearance"][side]["distance_m"],
        "arm_effort_fraction": arm_effort(row, side),
    }


def arm_effort(row, side):
    actuators = row["diagnostic_v1"]["actuators"]
    return max(
        a["effort_fraction"]
        for a in actuators.values()
        if a["joint"].startswith(side + "_")
        and "_o6_" not in a["joint"]
        and any(n in a["joint"] for n in ("shoulder", "elbow", "wrist"))
    )


def series(path):
    result = read(path / "summary.json")
    start = hold_entry(result)
    values = []
    with (path / "steps.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            values.append(
                {
                    "time_s": row["time"],
                    "hold_s": row["time"] - start if start is not None else None,
                    "phase": row["phase"],
                    "base_reset_m": row["base_drift_m"],
                    "base_hold_m": row["diagnostic_v1"]["from_hold"]["translation_m"],
                    "com_m": row["com"],
                    "base_m": row["base_pose"][:3],
                    "support_Fz_N": {
                        k: v["force_world"][2] for k, v in row["supports"].items()
                    },
                    "hands": {s: hand_signals(row, s) for s in ("left", "right")},
                }
            )
    return values


def matched_shares(rows):
    selected = [
        r
        for r in rows
        if r["hold_s"] is not None and 20 - 1e-8 <= r["hold_s"] <= 30 + 1e-8
    ]
    if len(selected) < 2 or selected[-1]["hold_s"] - selected[0]["hold_s"] < 9.97:
        return None
    return {
        s: float(np.mean([r["hands"][s]["share"] for r in selected]))
        for s in ("left", "right")
    }


def side_metrics(held, side):
    hands = [r["hands"][side] for r in held]
    if not hands:
        return {}
    position = np.array([h["palm_position_m"] for h in hands])
    target = np.array([h["thumb_target_rad"] for h in hands])
    result = {
        "mean_share": float(np.mean([h["share"] for h in hands])),
        "palm_migration_m": float(np.linalg.norm(position - position[0], axis=1).max()),
        "thumb_target_positive_advancement_rad": float(
            np.maximum(0, np.diff(target)).sum()
        ),
        "final_thumb_travel_rad": hands[-1]["thumb_travel_rad"],
        "final_thumb_work_J": hands[-1]["thumb_work_J"],
    }
    for key in (
        "thumb_normal_N",
        "finger_min_normal_N",
        "joint_margin_rad",
        "effort_margin_Nm",
        "housing_clearance_m",
    ):
        result["min_" + key] = min(h[key] for h in hands)
    for key in (
        "wrist_force_N",
        "wrist_moment_Nm",
        "wrist_bend_deg",
        "elbow_bend_deg",
        "arm_effort_fraction",
    ):
        result["max_" + key] = max(h[key] for h in hands)
    return result


def motion_cost(result):
    windows = result.get("drift_assessment", {}).get("windows", [])
    if not windows or windows[-1]["status"] == "NOT RUN":
        return None
    criteria = DriftCriteria()
    scores = []
    for signal in windows[-1]["signals"].values():
        p = signal["translation"]
        scores.extend(
            (
                p["slope_per_s"] / criteria.translation_slope_m_s,
                p["excursion"] / criteria.translation_excursion_m,
            )
        )
        if "rotation" in signal:
            r = signal["rotation"]
            scores.extend(
                (
                    r["slope_per_s"] / criteria.rotation_slope_rad_s,
                    r["excursion"] / criteria.rotation_excursion_rad,
                )
            )
    return max(scores)


def summarize_record(root, record):
    path = root / record["output"]
    result, meta = read(path / "summary.json"), read(path / "resolved.json")
    summary = {
        "candidate": record["candidate"],
        "stage": record["stage"],
        "path": record["output"],
        "configuration": meta["candidate"],
        "controller": meta["controller"],
        "support_targets": meta["support_targets"],
        "requested_s": meta["config"]["hold_seconds"],
        "strict": result["strict_status"],
        "result": result,
        "late_motion_cost": motion_cost(result),
        "reset": meta["reset_inspection"],
    }
    if result["strict_status"] == "NOT RUN":
        summary.update(achieved_s=0, hands={}, matched_shares=None)
        return summary
    rows = series(path)
    held = [r for r in rows if r["phase"] == "HOLD"]
    summary.update(
        achieved_s=result["strict"]["unsupported_hold_seconds"],
        hands={s: side_metrics(held, s) for s in ("left", "right")},
        matched_shares=matched_shares(held),
    )
    return summary


def rank(summary):
    result = summary["result"]
    quality = result.get("contact_quality", {})
    complete = summary["requested_s"] if summary["strict"] == "PASS" else 0
    cost = summary["late_motion_cost"]
    migration = max(
        quality.get("max_contact_centroid_migration_m", {}).values(),
        default=float("inf"),
    )
    return (
        not result["progression_justified"],
        -complete,
        -summary["achieved_s"],
        cost if cost is not None else float("inf"),
        quality.get("invalid_control_samples", 10**9),
        migration,
        result.get("strict", {}).get("peak_effort_fraction", float("inf")),
    )


def strict_contact_pass(summary):
    return (
        summary["strict"] == "PASS"
        and summary["result"].get("contact_quality", {}).get("status") == "PASS"
    )


def load_flag(shares, baseline):
    if shares is None or baseline is None:
        return None
    return {s: bool(shares[s] < 0.5 * baseline[s]) for s in baseline}


def campaign_results(campaign):
    completed = [r for r in campaign.state["runs"] if r["status"] == "COMPLETE"]
    results = [cached_summary(campaign.root, r) for r in completed]
    baseline = next(
        (
            r["matched_shares"]
            for r in results
            if r["candidate"] == "reference" and r["stage"] == "screen"
        ),
        None,
    )
    for result in results:
        result["materially_reduced_hand_load"] = load_flag(
            result["matched_shares"], baseline
        )
    return sorted(results, key=rank)


def cached_summary(root, record):
    source = root / record["output"] / "manifest.json"
    digest = hashlib.sha256(
        source.read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()
    cache = root / "comparison_cache" / (record["id"] + ".json")
    if cache.exists():
        value = read(cache)
        if value["source_and_analysis_sha256"] == digest:
            return value["summary"]
    result = summarize_record(root, record)
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(
        json.dumps({"source_and_analysis_sha256": digest, "summary": result}, indent=2)
        + "\n"
    )
    return result
