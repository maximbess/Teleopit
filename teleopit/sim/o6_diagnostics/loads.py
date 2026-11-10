"""Simultaneous signed demands; never assemble a wrench from component peaks."""

import numpy as np


def select_references(rows, side):
    eligible = [
        r
        for r in rows
        if r["phase"] == "HOLD"
        and r["assistance_gain"] == 0
        and not r.get("diagnostic_unscored", False)
    ]
    if not eligible:
        raise ValueError("source run has no zero-assistance HOLD samples")
    selections = [
        (
            "max_moment",
            max(
                eligible,
                key=lambda r: np.linalg.norm(r["wrist_world_on_hand"][side][3:]),
            ),
        ),
        (
            "max_force",
            max(
                eligible,
                key=lambda r: np.linalg.norm(r["wrist_world_on_hand"][side][:3]),
            ),
        ),
        ("final_pre_abort", eligible[-1]),
    ]
    selected = {}
    for label, row in selections:
        item = selected.setdefault(
            row["time"],
            {
                "selection": [],
                "time_s": row["time"],
                "side": side,
                "wrench_world_at_mount": row["wrist_world_on_hand"][side],
                "source_kind": "simulation_measured_demand_from_failed_or_passed_HOLD",
            },
        )
        item["selection"].append(label)
    return list(selected.values())
