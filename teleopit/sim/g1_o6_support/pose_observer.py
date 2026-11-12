"""Anatomical events and bounded detailed physics windows around first losses."""

import gzip
import json
from collections import deque
from contextlib import ExitStack

import numpy as np

from teleopit.sim.g1_o6.diagnostics import snapshot
from teleopit.sim.g1_o6.posture_report import annotate
from teleopit.sim.o6_diagnostics.io import write_json

from .audit import PhysicsAudit
from .pose_contacts import ContactEpisodes, contact_states, grasp_quality, hand_contacts
from .pose_motion import HandMotion, LadderClearance


class EventWindows:
    def __init__(self, path, dt):
        self.buffer = deque(maxlen=round(0.25 / dt) + 1)
        self.resources = ExitStack()
        self.stream = self.resources.enter_context(
            gzip.open(path / "physics_windows.jsonl.gz", "wt")  # noqa: SIM115 - owned by ExitStack
        )
        self.seen, self.triggers = set(), []
        self.until, self.last_written = -1, -1

    def add(self, sample, reasons):
        self.buffer.append(sample)
        new = set(reasons) - self.seen
        if new:
            self.triggers.append({"time_s": sample["time_s"], "signals": sorted(new)})
            self.seen.update(new)
            self.until = sample["time_s"] + 0.25
            for previous in self.buffer:
                self.write(previous)
        if sample["time_s"] <= self.until:
            self.write(sample)

    def write(self, sample):
        if sample["time_s"] > self.last_written:
            self.stream.write(json.dumps(sample) + "\n")
            self.last_written = sample["time_s"]


class PoseObserver(PhysicsAudit):
    def __init__(self, env, path):
        super().__init__(env)
        self.path = path
        self.motion = {s: HandMotion(env, s) for s in env.hands}
        self.clearance = LadderClearance(env)
        self.history = ContactEpisodes(env.config.physics_dt)
        self.hold_history = ContactEpisodes(env.config.physics_dt)
        self.windows = EventWindows(path, env.config.physics_dt)
        self.control_valid = []
        self.minimum_normal = {
            s: {"thumb": float("inf"), "finger": float("inf")} for s in env.hands
        }
        self.centroids, self.migration = {}, {}

    def hands(self, env, row):
        result = {}
        for side in env.hands:
            contacts = hand_contacts(env, row["contacts"], side)
            height = env.data.geom(env.support_targets[side + "_hand"]).xpos[2]
            result[side] = {
                "contacts": contacts,
                "quality": grasp_quality(contacts, height),
            }
        return result

    def __call__(self, env):
        row = super().__call__(env)
        hands = self.hands(env, row)
        states = contact_states(hands)
        self.history.update(row["time"], states)
        for motion in self.motion.values():
            motion.observe(env)
        reasons = self.physics_reasons(env, row, hands, states)
        sample = {
            "time_s": row["time"],
            "phase": env.phase,
            "hands": hands,
            "motion": {s: m.sample(env) for s, m in self.motion.items()},
            "base_pose": row["base_pose"],
            "base_drift_m": row["base_drift_m"],
            "wrist_world_on_hand": row["wrist_world_on_hand"],
            "actual_qfrc_applied": env.data.qfrc_applied.tolist(),
            "actual_xfrc_applied": env.data.xfrc_applied.tolist(),
        }
        self.windows.add(sample, reasons)
        return row

    def physics_reasons(self, env, row, hands, states):
        if env.phase != "HOLD":
            return []
        changed = self.hold_history.update(row["time"], states)
        self.track_contacts(hands)
        reasons = [
            n
            for n in changed
            if not states[n] and ("thumb" in n or n.endswith(("ordered", "valid")))
        ]
        reasons.extend(self.events.first)
        return ["hold_entry", *reasons]

    def track_contacts(self, hands):
        for side, hand in hands.items():
            quality = hand["quality"]
            for key, field in (
                ("thumb", "thumb_normal_N"),
                ("finger", "minimum_finger_normal_N"),
            ):
                self.minimum_normal[side][key] = min(
                    self.minimum_normal[side][key], quality[field]
                )
            for group, position in quality["centroid_by_group_m"].items():
                if position is not None:
                    name = side + ":" + group
                    initial = self.centroids.setdefault(name, position)
                    movement = float(np.linalg.norm(np.array(position) - initial))
                    self.migration[name] = max(self.migration.get(name, 0), movement)

    def control_sample(self, env, row=None):
        row = row if row is not None else annotate(env, snapshot(env))
        hands = self.hands(env, row)
        if env.phase == "HOLD":
            for motion in self.motion.values():
                motion.start_hold()
        for side, hand in hands.items():
            hand["motion"] = self.motion[side].control_sample(env)
            hand["clearance_by_obstacle_m"] = self.clearance.sample(env, side)
        if row["phase"] == "HOLD":
            self.control_valid.append(
                all(h["quality"]["valid"] for h in hands.values())
            )
        return {
            "hands": hands,
            "controller_state": getattr(env.hold_controller, "state", dict)(),
        }

    def enrich(self, env, row):
        row["pose_search"] = self.control_sample(env, row)

    def contact_report(self):
        absence = self.hold_history.longest_absence()
        persistent = {
            n: duration
            for n, duration in absence.items()
            if n.endswith(("thumb_opposition", "ordered", "valid"))
            and duration >= 0.5 - 1e-9
        }
        passed = bool(self.control_valid) and all(self.control_valid) and not persistent
        minima = {
            s: {k: v if np.isfinite(v) else None for k, v in values.items()}
            for s, values in self.minimum_normal.items()
        }
        return {
            "status": "PASS" if passed else "FAIL",
            "loaded_threshold_N": 0.05,
            "control_samples": len(self.control_valid),
            "invalid_control_samples": self.control_valid.count(False),
            "longest_absence_s": absence,
            "persistent_quality_losses_s": persistent,
            "minimum_group_normal_N": minima,
            "max_contact_centroid_migration_m": self.migration,
        }

    def close(self):
        self.windows.resources.close()
        write_json(
            self.path / "contact_episodes.json",
            {"all_phases": self.history.rows, "hold": self.hold_history.rows},
        )
        write_json(self.path / "physics_window_triggers.json", self.windows.triggers)
