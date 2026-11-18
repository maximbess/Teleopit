"""Bounded G6 ground-reaction refinement with native physics-step force gates."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from .g6_acquisition_audit import AcquisitionObserver
from .g6_campaign import STAGES
from .g6_ground_reaction import BALANCE_LIMIT_N, FINGER_RESERVE_N, GUARDS_N, TRIMS_N, GroundReaction
from .g6_trial import run_candidate
from .g6_whole_body_experiment import WholeBodyCampaign, candidates as body_candidates, eligible, main, measured_actors

PARENT_TRACE = "04c8bc7c11e4f6c1be37558aa77e2fd308efc61137e5a817163a0e9183c268ed"
PAIRS = {"collective": ("collective_16", "collective_21"),
         "individual": ("individual_16", "individual_21")}


def candidates():
    parent = next(c for c in body_candidates() if c.id == "wrench_high")
    parent = replace(parent, id="control", family="control", parent=None)
    return [parent, *[replace(parent, id=name, parent="control", family=family,
                              controller=GroundReaction(governor=family, guard_N=guard))
                      for family, names in PAIRS.items() for name, guard in zip(names, GUARDS_N)]]


def specification():
    return {"schema": "g6_ground_reaction_v1", "geometry": "G6", "native_mu": 1.8,
            "parent_probe_steps_sha256": PARENT_TRACE, "candidate_limit": 5,
            "candidates": {c.id: c.record() for c in candidates()}, "pairs": PAIRS,
            "stages": STAGES, "budgets": {s: (5 if s == "probe" else 4) for s in STAGES},
            "vertical_trim_choices_N": TRIMS_N, "finger_reserve_N": FINGER_RESERVE_N,
            "qualification": "native steps and control endpoints strictly below 25 N throughout HOLD, every left finger loaded on every native step, all original contact, physical and long convergence gates"}


class BalanceTrace:
    def __init__(self):
        self.samples, self.peak_N, self.violations = 0, 0., 0
        self.worst, self.first = None, None

    def observe(self, row, elapsed):
        self.samples += 1
        net = row["static_balance_residual_N"]
        magnitude = float(np.linalg.norm(net))
        exceeded = magnitude >= BALANCE_LIMIT_N
        self.violations += int(exceeded)
        if magnitude > self.peak_N or (exceeded and self.first is None):
            witness = {"hold_s": elapsed, "residual_world_N": net,
                       "supports_world_N": {s: h["force_world"] for s, h in row["supports"].items()}}
            if magnitude > self.peak_N:
                self.peak_N, self.worst = magnitude, witness
            if exceeded and self.first is None:
                self.first = witness


class GroundReactionObserver(AcquisitionObserver):
    def __init__(self, env, path):
        super().__init__(env, path)
        self.physical_balance, self.control_balance = BalanceTrace(), BalanceTrace()

    def physics_reasons(self, env, row, hands, states):
        if env.phase == "HOLD":
            self.physical_balance.observe(row, row["time"] - env.phase_start)
        return super().physics_reasons(env, row, hands, states)

    def enrich(self, env, row):
        super().enrich(env, row)
        if row["phase"] == "HOLD":
            self.control_balance.observe(row, row["hold_time_s"])

    def report(self):
        return {**super().report(), "ground_reaction": {
            "physics": vars(self.physical_balance), "control": vars(self.control_balance)}}


def run_refinement(payload, output, seconds, dt):
    return run_candidate(payload, output, seconds, dt, observer_factory=GroundReactionObserver)


def passes(result, stage):
    audit = result.get("physics_audit", {})
    control = audit.get("ground_reaction", {}).get("control", {})
    peak = audit.get("extrema", {}).get("force_balance", float("inf"))
    return (control.get("samples", 0) > 0 and control.get("peak_N", float("inf")) < BALANCE_LIMIT_N
            and peak < BALANCE_LIMIT_N and eligible(result, stage))


def contact_actors(record):
    return {**measured_actors(record), "balance": record["result"]["physics_audit"]["ground_reaction"]}


class GroundReactionCampaign(WholeBodyCampaign):
    trial_function = staticmethod(run_refinement)
    spec_factory = staticmethod(specification)
    passes = staticmethod(passes)
    actors = staticmethod(contact_actors)
    first_pair, second_pair = PAIRS.values()
    audit_key = "collective_premise"
    premise = "A common vertical trim and a pelvis moment chosen from posture are sufficient for both feet."
    replacement = "Allow the same bounded vertical trim at either foot, including its lever-arm moment in the constrained pelvis wrench."
    module = "teleopit.sim.g1_o6_support.g6_ground_reaction_experiment"


if __name__ == "__main__":
    main(GroundReactionCampaign, Path("outputs/g1_o6/g6_ground_reaction"))
