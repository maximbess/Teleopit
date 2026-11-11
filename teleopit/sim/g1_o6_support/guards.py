"""Pure measured transition predicates; clocks are supplied by the simulator."""

import numpy as np

from .config import TransferCriteria

CRITERIA = TransferCriteria()


class Persistence:
    def __init__(self, seconds):
        self.seconds = seconds
        self.since = None

    def update(self, time, predicate):
        if not predicate:
            self.since = None
        elif self.since is None:
            self.since = time
        return self.since is not None and time - self.since >= self.seconds - 1e-9


def vertical_reactions(row):
    return {n: v["force_world"][2] for n, v in row["supports"].items()}


def unloading(reference, row, side, cfg=CRITERIA):
    before, now = vertical_reactions(reference), vertical_reactions(row)
    name = side + "_hand"
    initial, current = before[name], now[name]
    lost = initial - current
    gained = sum(now[n] - before[n] for n in before if n != name)
    wrench = row["wrist_world_on_hand"][side]
    checks = {
        "meaningful_initial_load": initial >= cfg.unload_floor_N,
        "reduction": abs(current) <= (1 - cfg.unload_reduction) * initial,
        "wrist_force": np.linalg.norm(wrench[:3]) < cfg.wrist_force_N,
        "wrist_moment": np.linalg.norm(wrench[3:]) < cfg.wrist_moment_Nm,
        "reaction_compensation": gained > 0
        and abs(lost - gained) <= cfg.reaction_tolerance_N,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": {k: bool(v) for k, v in checks.items()},
        "initial_Fz_N": initial,
        "selected_Fz_N": current,
        "lost_Fz_N": lost,
        "other_support_gain_N": gained,
    }


def share_settled(row, target, weight, cfg=CRITERIA):
    shares = [
        row["supports"][s + "_hand"]["force_world"][2] / weight
        for s in ("left", "right")
    ]
    return abs(sum(shares) - target) <= cfg.share_tolerance and all(
        cfg.minimum_per_hand_share <= v <= cfg.maximum_per_hand_share for v in shares
    )


def released(start, actual, contacts, clearance, cfg=CRITERIA):
    return bool(
        np.max(np.abs(actual)) <= cfg.open_tolerance_rad
        and np.max(np.asarray(start) - actual) >= cfg.minimum_open_motion_rad
        and contacts == 0
        and clearance >= cfg.clearance_m
    )


def wrench_covered(demand, passed_wrenches, reserve=1.2):
    """Only tested positive rays cover a demand; norms alone prove nothing."""
    demand = np.asarray(demand, float)
    norm2 = demand @ demand
    if norm2 < 1e-20:
        return any(np.linalg.norm(w) < 1e-10 for w in passed_wrenches)
    for wrench in passed_wrenches:
        scale = np.dot(wrench, demand) / norm2
        if scale >= reserve - 1e-9 and np.allclose(
            wrench, scale * demand, atol=1e-8, rtol=1e-7
        ):
            return True
    return False
