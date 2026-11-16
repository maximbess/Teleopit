"""Continuous measured load and phase-local convergence, without resetting poses."""

import numpy as np

from .measurements import assess_window


def phase_convergence(rows, phase, side=None):
    selected = [
        r
        for r in rows
        if r.get("experiment_phase") == phase
        and (side is None or r["transfer"]["side"] == side)
    ]
    if not selected:
        return {"status": "NOT RUN", "signals": {}}
    local = [{**r, "hold_time_s": r["experiment_phase_time_s"]} for r in selected]
    end = local[-1]["hold_time_s"]
    if end < 29.98:
        return {"status": "NOT RUN", "signals": {}}
    return assess_window(local, end - 30, end)


def load_measurements(rows, weight, dt):
    selected = [r for r in rows if r.get("experiment_phase") == "LOADED_HOLD"]
    if not selected:
        return {"status": "NOT RUN", "continuous_in_band_s": 0}
    shares = measured_shares(selected, weight)
    valid = [r["transfer"]["measured"]["share"] for r in selected]
    duration = selected[-1]["time"] - selected[0]["time"] + dt
    return {
        "status": "PASS" if all(valid) else "FAIL",
        "continuous_in_band_s": duration if all(valid) else 0,
        "fraction_samples_in_band": sum(valid) / len(valid),
        "samples": len(valid),
        "total_share_min_max": [
            float(shares.sum(axis=1).min()),
            float(shares.sum(axis=1).max()),
        ],
        "per_hand_min_max": {
            s: [float(shares[:, i].min()), float(shares[:, i].max())]
            for i, s in enumerate(("left", "right"))
        },
    }


def measured_shares(rows, weight):
    return np.array(
        [
            [
                r["supports"][s + "_hand"]["force_world"][2] / weight
                for s in ("left", "right")
            ]
            for r in rows
        ]
    )


def transfer_motion_gate(request, rows):
    plateau = phase_convergence(rows, "LOADED_HOLD")
    destination = {
        s: phase_convergence(rows, "POST_HOLD", s) for s in ("left", "right")
    }
    required = []
    if request["plateau_seconds"] >= 60:
        required.append(plateau["status"] == "PASS")
    if request["mode"].startswith("adjacent"):
        sides = (
            (request["side"],)
            if request["mode"] == "adjacent-first"
            else ("left", "right")
        )
        required.extend(destination[s]["status"] == "PASS" for s in sides)
    return {
        "passed": all(required),
        "plateau_convergence": plateau,
        "destination_convergence": destination,
    }


def capacity_share_bounds(proof, weight):
    if proof is None:
        return None
    return {
        s: max([0, *(-w[2] / (1.2 * weight) for w in proof.passed_wrenches[s])])
        for s in ("left", "right")
    }


def adjacent_progress(request, events, physical, covered):
    if not request["mode"].startswith("adjacent"):
        return {"status": "NOT RUN"}
    completed = {e["side"] for e in events if e["previous"] == "POST_HOLD"}
    first = request["side"]
    second = "right" if first == "left" else "left"
    return {
        "first_hand": "PASS" if first in completed else "FAIL",
        "second_hand": second_hand_status(request["mode"], first, second, completed),
        "complete_sequence": "PASS"
        if request["mode"] == "adjacent" and physical and covered
        else "BLOCKED",
        "completed_hands": sorted(completed),
    }


def second_hand_status(mode, first, second, completed):
    if mode == "adjacent-first":
        return "NOT RUN"
    if second in completed:
        return "PASS"
    return "BLOCKED" if first not in completed else "FAIL"
