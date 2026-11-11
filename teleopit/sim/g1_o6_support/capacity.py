"""Fresh free-wrist trials with explicit ramps, recovery and sampled source state."""

import hashlib
import json
from dataclasses import dataclass

import mujoco
import numpy as np

from teleopit.sim.o6_diagnostics.config import DiagnosticConfig
from teleopit.sim.o6_diagnostics.hand_results import SCORED_PHASES
from teleopit.sim.o6_diagnostics.hand_run import HandTrial, run_hand
from teleopit.sim.o6_diagnostics.harness import Source
from teleopit.sim.o6_diagnostics.io import read_rows
from teleopit.sim.o6_diagnostics.loads import select_references
from teleopit.sim.o6_grasp.controller import smooth


@dataclass(frozen=True)
class CapacityConfig(DiagnosticConfig):
    load_ramp_s: float = 3.0
    ramp_down_s: float = 3.0
    recovery_s: float = 10.0


def capacity_durations(cfg):
    return [
        cfg.acquire_open_s,
        cfg.acquire_close_s,
        cfg.fixture_remove_s,
        cfg.zero_fixture_settle_s,
        cfg.load_ramp_s,
        cfg.plateau_s,
        cfg.ramp_down_s,
        cfg.recovery_s,
    ]


def capacity_schedule(t, cfg):
    durations = capacity_durations(cfg)
    ends = np.cumsum(durations)
    i = min(int(np.searchsorted(ends, t + 1e-9, side="right")), 7)
    start = ends[i - 1] if i else 0
    fraction = smooth((t - start) / durations[i])
    phase = [
        "OPEN",
        "CLOSE",
        "REMOVE_FIXTURE",
        "SETTLE",
        "LOAD_RAMP",
        "PLATEAU",
        "RAMP_DOWN",
        "RECOVERY",
    ][i]
    amount = 0 if i == 0 else fraction if i == 1 else 1
    gain = 1 if i < 2 else 1 - fraction if i == 2 else 0
    load = {4: fraction, 5: 1, 6: 1 - fraction}.get(i, 0)
    return phase, amount, gain, load


class CapacityTrial(HandTrial):
    def __init__(self, h, cfg, wrench, protocol, only):
        super().__init__(h, cfg, wrench, protocol, only)
        self.schedule = self.capacity_schedule
        self.end = sum(capacity_durations(cfg))

    def capacity_schedule(self, time, cfg):
        phase, amount, gain, load = capacity_schedule(time, cfg)
        if self.h.source_sample is not None:
            amount = 1.0
        return phase, amount, gain, load

    def physics_step(self, stream):
        running = super().physics_step(stream)
        if self.phase in SCORED_PHASES:
            self.audit_applied_load()
        return running

    def audit_applied_load(self):
        h = self.h
        expected = np.zeros(h.model.nv)
        mujoco.mj_applyFT(
            h.model,
            h.data,
            h.external[:3],
            h.external[3:],
            h.data.site(h.hand["mount"]).xpos,
            h.base,
            expected,
        )
        if np.any(h.fixture):
            self.failure.setdefault("fixture_support", self.clock)
        if not np.allclose(h.data.qfrc_applied, expected, rtol=0, atol=1e-12) or np.any(
            h.data.xfrc_applied
        ):
            self.failure.setdefault("undeclared_applied_force", self.clock)


def measured_references(rows, side):
    references = select_references(rows, side)
    for reference in references:
        reference["selection"] = [
            "settled" if label == "final_pre_abort" else label
            for label in reference["selection"]
        ]
    return references


def qualified_source(path, state="initial"):
    source = Source(path)
    summary = json.loads((source.path / "summary.json").read_text())
    strict = summary.get("strict", summary.get("original_strict_outcome", summary))
    complete = strict.get("final_phase") == "HOLD_COMPLETE"
    passed = (
        strict.get("static_hold_passed", False)
        or summary.get("physical_status") == "PASS"
    )
    if not complete or not passed:
        raise ValueError("capacity source must contain a qualified coupled hold")
    if state == "settled":
        rows = read_rows(source.path / "steps.jsonl")
        row = next(r for r in reversed(rows) if r["phase"] == "HOLD")
        source.data.qpos[:] = [*row["base_pose"], *row["joint_positions"]]
        mujoco.mj_forward(source.model, source.data)
        source.sample_reference = {
            "time_s": row["time"],
            "frame": "world",
            "selection": "last strictly qualified HOLD control sample",
            "row_sha256": hashlib.sha256(
                json.dumps(row, sort_keys=True).encode()
            ).hexdigest(),
            "initialization": "sampled mount and hand joint configuration; reset velocities zero; motor preload initialized",
        }
    elif state != "initial":
        raise ValueError("source state must be initial or settled")
    return source


def run_capacity(source, side, output, *, wrench=None, physics_dt=None, reference=None):
    value = np.zeros(6) if wrench is None else np.asarray(wrench)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("expected six finite load components")
    if np.linalg.norm(value[:3]) > 50 or np.linalg.norm(value[3:]) > 3:
        raise ValueError("test wrench exceeds the predeclared 50 N / 3 Nm budget")
    return run_hand(
        source,
        side,
        output,
        wrench=wrench,
        dt=physics_dt,
        config=CapacityConfig(),
        protocol="support_capacity_v1",
        trial_factory=CapacityTrial,
        load_reference=reference
        or {"kind": "gravity_only", "axes": "world", "origin": "wrist_mount"},
    )
