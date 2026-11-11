"""Read-only evidence for the opt-in balanced grasp, at control-sample resolution."""

from .posture import posture_metrics


def annotate(env, row):
    row["posture"] = posture_metrics(env)
    row["posture"]["thumb_below_fingers_above"] = grasp_ordering(env, row)
    row["applied_position_targets_rad"] = env.data.ctrl.tolist()
    return row


def grasp_ordering(env, row):
    return {
        side: ordered_contacts(
            row["contacts"], side, target, env.data.geom(target).xpos[2]
        )
        for side in env.hands
        for target in [env.support_targets[side + "_hand"]]
    }


def ordered_contacts(contacts, side, target, height):
    selected = [
        c
        for c in contacts
        if target in c["geoms"]
        and side + "_hand" in c["regions"]
        and c["normal_force"] >= 0.05
    ]
    return all(
        any(
            any("_" + finger + "_" in g for g in c["geoms"])
            and direction * (c["position"][2] - height) > 0
            for c in selected
        )
        for finger, direction in (
            ("thumb", -1),
            ("index", 1),
            ("middle", 1),
            ("ring", 1),
            ("pinky", 1),
        )
    )


def housing_contact(row):
    return any(
        "environment" in c["regions"]
        and c["normal_force"] > 0.05
        and any("_wrist_" in b for b in c["bodies"])
        for c in row["contacts"]
    )


def summarize_posture(rows, control_dt):
    held = [r["posture"] for r in rows if r["phase"] == "HOLD"]
    return {
        "sampling_interval_s": control_dt,
        "housing_distance_cap_m": 0.1,
        "initial": rows[0]["posture"],
        "final": rows[-1]["posture"],
        "housing_contact_samples": sum(housing_contact(r) for r in rows),
        "hold_thumb_below_fingers_above": bool(held)
        and all(all(p["thumb_below_fingers_above"].values()) for p in held),
        "max_hold_wrist_bend_deg": max(
            (max(p["wrist_bend_deg"].values()) for p in held), default=None
        ),
        "min_hold_housing_clearance_m": min(
            (v["distance_m"] for p in held for v in p["housing_clearance"].values()),
            default=None,
        ),
    }
