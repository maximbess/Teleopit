"""Reconstruct bilateral opposition and adaptive grip commands from native G6 evidence."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.g1_o6_g6_acquisition_report import opposed_contacts, selected_contacts
from scripts.verify_g1_o6_g6_knee import verify_run as verify_knee
from scripts.verify_g1_o6_g6_knee_retention import GripAudit, equal, verify_grip
from scripts.verify_g1_o6_g6_whole_body import read, verify
from teleopit.sim.g1_o6_support.pose_contacts import FINGERS

SIDES = ("left", "right")


def grasp_sample(row, side):
    groups = {digit: [] for digit in FINGERS}
    for contact in selected_contacts(row, side):
        if contact["group"] in groups:
            groups[contact["group"]].append(contact)
    loads = np.array([sum(c["force"] for c in group) for group in groups.values()])
    return loads, opposed_contacts(groups)


class ContactEvidence:
    def __init__(self):
        self.count = 0
        self.minima = np.full((2, 5), np.inf)
        self.losses = dict.fromkeys(("left_thumb", "right_thumb", "left_fingers"), 0)
        self.first = dict.fromkeys(self.losses)

    def observe(self, row, elapsed):
        samples = [grasp_sample(row, side) for side in SIDES]
        forces, opposed = zip(*samples)
        self.minima = np.minimum(self.minima, forces)
        self.count += 1
        loaded = {s + "_thumb": has_opposition for s, has_opposition in zip(SIDES, opposed)}
        loaded["left_fingers"] = min(forces[0][1:]) >= .05
        for name, present in loaded.items():
            if not present:
                self.losses[name] += 1
                if self.first[name] is None:
                    self.first[name] = elapsed

    def verify(self, saved):
        assert self.count > 0 and self.count == saved["samples"] and self.losses == saved["loss_steps"]
        for key, at in self.first.items():
            claimed = saved["first_loss_s"][key]
            assert claimed is None if at is None else abs(at - claimed) < 1e-8
        return {**saved, "minimum_N": {s: dict(zip(FINGERS, values.tolist())) for s, values in zip(SIDES, self.minima)}}


def contacts_at_both_rates(path, meta, reported):
    native, control = ContactEvidence(), ContactEvidence()
    with gzip.open(path / "physics_contacts.jsonl.gz", "rt") as stream:
        for row in map(json.loads, stream):
            if row["phase"] == "HOLD":
                native.observe(row, row["phase_time_s"])
    with (path / "steps.jsonl").open() as stream:
        for row in map(json.loads, stream):
            if row["phase"] == "HOLD":
                row["support_targets"] = meta["support_targets"]
                control.observe(row, row["hold_time_s"])
    return {"native": native.verify(reported["native"]), "control": control.verify(reported["control"])}


def desired_rate(load, target, gain):
    error = target - load
    return gain * (np.maximum(error - .08, 0.) + np.minimum(error + .08, 0.))


def check_cap(saved, last, anchor, native, loads, opposed, height_error, dt, spec, inhibited):
    gain = spec["pressure_gain_rad_per_Ns"]
    rate = np.zeros_like(last)
    rate[:, 0] = desired_rate(np.where(opposed, loads[:, 0], 0.), .70, gain)
    if spec["law"] == "palm":
        rate[:, 2:] = 6. * height_error[:, None] + gain * np.maximum(1. - loads[:, 1:], 0.)
    else:
        rate[:, 2:] = desired_rate(loads[:, 1:], 1.40, gain)
    rate[(last > native) & (rate > 0.)] = 0.
    change = np.clip(rate, -.04, .04) * dt
    expected = last if inhibited else np.clip(last + change, anchor - .12, anchor + .12)
    if not equal(saved, expected):
        raise ValueError("feedback cap differs from measured contacts and bounded servo response")


class FeedbackAudit(GripAudit):
    def __init__(self, model, meta):
        super().__init__(model, meta)
        self.spec = meta["controller"]["grip_feedback"]
        assert (self.spec["law"], self.spec["pressure_gain_rad_per_Ns"]) in (("pressure", .04), ("pressure", .12), ("palm", .12))
        self.dt = meta["config"]["control_dt"]
        self.targets = meta["support_targets"]
        self.anchor_height = self.previous_cap = None
        self.peak_travel = self.peak_speed = 0.

    def feedback_cap(self, previous, control, native):
        anchor = control["settled_grip"]["ceiling_rad"]
        saved = control["grip_feedback_cap_rad"]
        if anchor is None:
            assert saved is None
            return None
        anchor = np.asarray(anchor).reshape(2, 6)
        cap = np.asarray(saved).reshape(2, 6)
        heights = np.array([previous["hand_in_target_rung"][s]["position"][2] for s in SIDES])
        if self.previous_cap is None:
            self.previous_cap, self.anchor_height = anchor, heights
        raw = {"support_targets": self.targets, "contacts": previous["contacts"]}
        loads, opposed = zip(*(grasp_sample(raw, s) for s in SIDES))
        check_cap(cap, self.previous_cap, anchor, native.reshape(2, 6), np.asarray(loads),
                  opposed, self.anchor_height-heights, self.dt, self.spec, control["inhibition"])
        self.peak_travel = max(self.peak_travel, float(np.max(abs(cap-anchor))))
        self.peak_speed = max(self.peak_speed, float(np.max(abs(cap-self.previous_cap))) / self.dt)
        self.previous_cap = cap
        return cap.ravel()

    def result(self):
        return {**super().result(), "peak_feedback_travel_rad": self.peak_travel,
                "peak_feedback_rate_rad_s": self.peak_speed}


def verify_run(path, report):
    checked = verify_knee(path, report)
    meta, audit = read(path / "resolved.json"), read(path / "summary.json")["physics_audit"]
    kind = FeedbackAudit if "grip_feedback" in meta["controller"] else GripAudit
    commands = verify_grip(path, meta, kind)
    assert commands["capture_s"] == audit["settled_grip"]["capture_s"]
    return {**checked, "grip_support": contacts_at_both_rates(path, meta, audit["grip_support"]),
            "grip_commands": commands}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/g1_o6/g6_grip_feedback"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    verify(args.root, args.run_id, verify_function=verify_run)
