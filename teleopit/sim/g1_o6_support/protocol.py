"""An explicit finite protocol; transitions depend on measured predicates."""

from dataclasses import dataclass

from .config import TransferCriteria
from .guards import Persistence


@dataclass(frozen=True)
class Step:
    name: str
    minimum_s: float
    timeout_s: float
    guard: str = "tick"  # Physical gates retain their own persistence in the runner.
    side: str = "left"
    hold: bool = False


def transfer_steps(mode, side, cycles, alternating, three_seconds, plateau_seconds):
    cfg = TransferCriteria()
    steps = [
        Step("BASELINE", cfg.reference_s, cfg.reference_s, hold=True),
        Step("LOAD_RAMP", cfg.ramp_s, cfg.ramp_s),
        Step("LOAD_SETTLE", 0, cfg.settle_timeout_s, "share"),
        Step("LOADED_HOLD", plateau_seconds, plateau_seconds, "share", hold=True),
    ]
    if mode != "load":
        for i in range(cycles):
            selected = (
                ("right" if side == "left" else "left")
                if alternating and i % 2
                else side
            )
            steps.extend(cycle_steps(mode, selected, three_seconds, cfg))
    if mode != "release":
        steps.extend(
            [
                Step("RETURN", cfg.ramp_s, cfg.ramp_s),
                Step(
                    "RECOVERY",
                    cfg.recovery_s,
                    cfg.recovery_s,
                    "baseline_share",
                    hold=True,
                ),
            ]
        )
    return steps


def cycle_steps(mode, side, three_seconds, cfg):
    steps = [
        Step("UNLOAD_RAMP", cfg.ramp_s, cfg.ramp_s, side=side),
        Step("UNLOAD_SETTLE", 0, cfg.settle_timeout_s, "unloaded", side),
        Step(
            "UNLOADED_HOLD",
            cfg.unloaded_hold_s,
            cfg.unloaded_hold_s,
            "unloaded",
            side,
            True,
        ),
    ]
    if mode in ("release", "reacquire"):
        steps.extend(
            [
                Step("OPEN", cfg.opening_s, cfg.settle_timeout_s, "opened", side),
                Step(
                    "RETRACT", cfg.trajectory_s, cfg.settle_timeout_s, "detached", side
                ),
                Step(
                    "THREE_SUPPORT_HOLD",
                    three_seconds,
                    three_seconds,
                    "detached",
                    side,
                    True,
                ),
            ]
        )
    if mode == "reacquire":
        steps.extend(
            [
                Step(
                    "APPROACH",
                    cfg.trajectory_s,
                    cfg.settle_timeout_s,
                    "approached",
                    side,
                ),
                Step("CLOSE", cfg.opening_s, cfg.settle_timeout_s, "grasp", side),
                Step(
                    "VERIFY_GRASP",
                    cfg.persistence_s,
                    cfg.settle_timeout_s,
                    "grasp",
                    side,
                ),
            ]
        )
    if mode != "release":
        steps.extend(
            [
                Step("RELOAD", cfg.ramp_s, cfg.ramp_s, side=side),
                Step("RELOAD_SETTLE", 0, cfg.settle_timeout_s, "accepted_load", side),
                Step(
                    "POST_HOLD",
                    cfg.plateau_s,
                    cfg.plateau_s,
                    "accepted_load",
                    side,
                    True,
                ),
            ]
        )
    return steps


class Protocol:
    def __init__(self, steps, persistence_s=0.5):
        if not steps:
            raise ValueError("empty protocol")
        self.steps = steps
        self.index = 0
        self.started = None
        self.guard = Persistence(persistence_s)
        self.failure = None
        self.events = []

    @property
    def phase(self):
        if self.failure:
            return "ABORT"
        return "COMPLETE" if self.index == len(self.steps) else self.step.name

    @property
    def step(self):
        return self.steps[self.index]

    def elapsed(self, time):
        return 0 if self.started is None else time - self.started

    def update(self, time, measured):
        if self.phase in ("ABORT", "COMPLETE"):
            raise RuntimeError("protocol finished")
        if self.started is None:
            self.started = time
        step = self.step
        valid = bool(measured.get(step.guard, False))
        persistent = self.guard.update(time, valid)
        elapsed = self.elapsed(time)
        if step.hold and not valid:
            self.failure = f"{step.name}:interrupted:{step.guard}"
        elif elapsed >= step.minimum_s - 1e-9 and persistent:
            self.index += 1
            self.events.append(
                {
                    "time": time,
                    "previous": step.name,
                    "next": self.phase,
                    "side": step.side,
                }
            )
            self.started = time
            self.guard.since = None
        elif elapsed >= step.timeout_s - 1e-9:
            self.failure = f"{step.name}:timeout:{step.guard}"
