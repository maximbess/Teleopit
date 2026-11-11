"""Bounded sampled capacity; a failed zero-fixture settle blocks later loading."""

from functools import partial

import numpy as np

from teleopit.sim.o6_diagnostics.hand_run import run_hand
from teleopit.sim.o6_diagnostics.io import (
    manifest,
    output_directory,
    read_rows,
    write_json,
)

from .capacity import (
    CapacityConfig,
    CapacityTrial,
    measured_references,
    qualified_source,
    run_capacity,
)


def load_cases(references):
    cases = []
    for i, reference in enumerate(references):
        measured = np.asarray(reference["wrench_world_at_mount"])
        for family, mask in (
            ("force", [1, 1, 1, 0, 0, 0]),
            ("moment", [0, 0, 0, 1, 1, 1]),
            ("combined", [1] * 6),
        ):
            for scale in (
                (0.5, 1, 1.2, 1.5, 2) if family == "combined" else (0.5, 1, 1.5, 2)
            ):
                wrench = scale * measured * mask
                if np.linalg.norm(wrench[:3]) <= 50 and np.linalg.norm(wrench[3:]) <= 3:
                    cases.append(
                        {
                            "name": f"sample_{i}_{family}_{scale:g}",
                            "family": family,
                            "scale": scale,
                            "wrench": wrench.tolist(),
                            "reference": reference,
                        }
                    )
    return cases


def trial_passes(result):
    return (
        result["strict_pass"]
        and result["plateau_complete"]
        and result["physical_retention"]
    )


def evaluate_side(source, side, path, references, dt, budget):
    zero = run_capacity(source, side, path / "zero", physics_dt=dt)
    result = {
        "side": side,
        "physical_status": "FAIL",
        "loading_status": "BLOCKED",
        "reserve_qualified": False,
        "trial_budget": budget,
        "trials": [{"name": "zero", "result": zero}],
    }
    if not trial_passes(zero):
        result["reason"] = (
            "Zero-load settling/retention failed; ramp, plateau and recovery cannot be qualified."
        )
        return result
    cases = load_cases(references)
    if len(cases) + 1 > budget:
        raise ValueError("capacity matrix exceeds the declared trial budget")
    for case in cases:
        outcome = run_capacity(
            source,
            side,
            path / case["name"],
            wrench=case["wrench"],
            physics_dt=dt,
            reference={
                **case,
                "axes": "world",
                "origin": "wrist_mount",
                "follower": False,
            },
        )
        result["trials"].append({**case, "result": outcome})
    result["loading_status"] = (
        "PASS" if all(trial_passes(c["result"]) for c in result["trials"]) else "FAIL"
    )
    reserves = [c for c in result["trials"] if c.get("scale") == 1.2]
    result["reserve_qualified"] = bool(reserves) and all(
        trial_passes(c["result"]) for c in reserves
    )
    result["physical_status"] = result["loading_status"]
    return result


class ReplayTrial(CapacityTrial):
    def __init__(self, *args, time, wrenches):
        super().__init__(*args)
        self.replay_time = np.asarray(time) - time[0]
        self.replay_wrenches = np.asarray(wrenches)

    def physics_step(self, stream):
        # Ramp to the first recorded sample, replay its original world axes
        # during the plateau, then ramp the final simultaneous sample to zero.
        plateau_start = sum(
            (
                self.cfg.acquire_open_s,
                self.cfg.acquire_close_s,
                self.cfg.fixture_remove_s,
                self.cfg.zero_fixture_settle_s,
                self.cfg.load_ramp_s,
            )
        )
        self.wrench = np.array(
            [
                np.interp(
                    self.clock - plateau_start,
                    self.replay_time,
                    self.replay_wrenches[:, j],
                )
                for j in range(6)
            ]
        )
        return super().physics_step(stream)


def run_replay(source, side, output, rows, dt):
    end = rows[-1]["time"]
    selected = [r for r in rows if end - 10 <= r["time"] <= end]
    time = [r["time"] for r in selected]
    wrenches = [r["wrist_world_on_hand"][side] for r in selected]
    values = np.asarray(wrenches)
    if (
        time[-1] - time[0] < 9.98
        or np.max(np.linalg.norm(values[:, :3], axis=1)) > 50
        or np.max(np.linalg.norm(values[:, 3:], axis=1)) > 3
    ):
        raise ValueError("replay needs 10 s within the declared load limits")
    return run_hand(
        source,
        side,
        output,
        dt=dt,
        config=CapacityConfig(),
        protocol="support_replay_v1",
        trial_factory=partial(ReplayTrial, time=time, wrenches=wrenches),
        load_reference={
            "axes": "world",
            "origin": "wrist_mount",
            "follower": False,
            "source_times_s": time,
            "simultaneous_wrenches_N_Nm": wrenches,
        },
    )


def run_matrix(source_path, output, *, state="settled", dt=0.001, prior_trials=0):
    source = qualified_source(source_path, state)
    path = output_directory(output)
    rows = read_rows(source.path / "steps.jsonl")
    result = {
        "schema": "g1_o6_capacity_matrix_v1",
        "source_state": state,
        "source_run": str(source.path),
        "sides": {},
    }
    for side in ("left", "right"):
        report = evaluate_side(
            source,
            side,
            path / side,
            measured_references(rows, side),
            dt,
            48 - prior_trials,
        )
        result["sides"][side] = report
        # Replay is explicitly ineligible when any static case failed.
        if (
            report["loading_status"] == "PASS"
            and len(report["trials"]) < report["trial_budget"]
        ):
            report["replay"] = run_replay(
                source,
                side,
                path / side / "replay",
                [r for r in rows if r["phase"] == "HOLD"],
                dt,
            )
        else:
            report["replay"] = {
                "physical_status": "BLOCKED",
                "reason": "static matrix incomplete or failed, or budget exhausted",
            }
        write_json(path / "summary.json", result)
    manifest(path, "support_capacity_matrix", {})
    return result
