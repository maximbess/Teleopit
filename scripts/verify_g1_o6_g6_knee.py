"""Reconstruct the executed G6 knee geometry, load, grip, force and motion gates."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from scripts.g1_o6_g6_acquisition_report import verify_anatomy
from scripts.verify_g1_o6_g6_ground_reaction import verify_run as verify_reaction
from scripts.verify_g1_o6_g6_whole_body import read, verify
from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.diagnostics import failure_reasons
from teleopit.sim.g1_o6.posture import geom_clearance
from teleopit.sim.g1_o6_support.g6_knee_experiment import read_knee_rows
from teleopit.sim.g1_o6_support.measurements import late_drift

RUNG = "left_ladder_rung_03"
TOLERANCE = 1e-9


def contact_loads(contacts, names):
    return [c["normal_force"] for c in contacts if RUNG in c["geoms"] and names.intersection(c["geoms"])]


def check_native_sample(record, row, distance, names):
    measured = distance(record[3:])
    loads = contact_loads(row["contacts"], names)
    if not abs(record[0] - row["time_s"]) <= TOLERANCE:
        raise ValueError("native knee and contact timestamps differ")
    if not abs(record[1] - measured) <= TOLERANCE or record[2] != len(loads):
        raise ValueError("measured knee geometry or raw pair contact differs")
    return measured, max(loads, default=0.)


def check_endpoint(row, native, gap):
    pose = [*row["base_pose"], *row["joint_positions"]]
    if not np.array_equal(pose, native[3:]) or not abs(row["time"]-native[0]) <= TOLERANCE:
        raise ValueError("control state is not the executed native interval endpoint")
    if not abs(gap-native[1]) <= TOLERANCE:
        raise ValueError("control knee geometry differs from final integrated pose")


class RightKneeGeometry:
    def __init__(self, model):
        self.model, self.data = model, mujoco.MjData(model)
        body = model.body("right_knee_link").id
        self.geoms = np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0)
                                    & (model.geom_pos[:, 2] >= -.1))
        self.rung = model.geom(RUNG).id
        self.names = {model.geom(g).name for g in self.geoms}
        if len(self.names) != 7:
            raise ValueError("expected seven authentic right-knee collision meshes")

    def distance(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return min(geom_clearance(self.model, self.data, g, self.rung)[0] for g in self.geoms)


def extrema(gaps, contact_counts, times, peak_load):
    index = int(np.argmin(gaps))
    touched = np.flatnonzero(contact_counts)
    return {"samples": len(gaps), "minimum_m": float(gaps[index]), "closest_s": float(times[index]),
            "contact_steps": len(touched), "first_contact_s": float(times[touched[0]]) if len(touched) else None,
            "peak_contact_N": float(peak_load)}


def verify_native(path, records, geometry, dt):
    if not len(records) or not np.all(np.abs(np.diff(records[:, 0])-dt) <= TOLERANCE):
        raise ValueError("missing or duplicated native knee samples")
    gaps, peak = [], 0.
    with gzip.open(path / "physics_contacts.jsonl.gz", "rt") as stream:
        held = (r for r in map(json.loads, stream) if r["phase"] == "HOLD")
        for record, row in zip(records, held, strict=True):
            measured, load = check_native_sample(record, row, geometry.distance, geometry.names)
            gaps.append(measured)
            peak = max(peak, load)
    return extrema(gaps, records[:, 2], records[:, 0]-records[0, 0]+dt, peak)


def verify_controls(path, records, geometry, meta):
    cfg = Config(**meta["config"])
    interval = round(cfg.control_dt/cfg.physics_dt)
    endpoints = records[interval-1::interval]
    if len(endpoints)*interval != len(records):
        raise ValueError("incomplete control interval")
    poses, counts, gaps, times = [], [], [], []
    reasons, peak = Counter(), 0.
    with (path / "steps.jsonl").open() as stream:
        held = (r for r in map(json.loads, stream) if r["phase"] == "HOLD")
        for row, endpoint in zip(held, endpoints, strict=True):
            gap = row["g6"]["body_clearance"]["right_knee"][RUNG]["signed_distance_m"]
            check_endpoint(row, endpoint, gap)
            loads = contact_loads(row["contacts"], geometry.names)
            actual = failure_reasons(row, cfg, list(meta["support_targets"]))
            assert actual == row["failure_reasons"]
            reasons.update(actual)
            peak = max(peak, max(loads, default=0.))
            times.append(row["hold_time_s"])
            counts.append(len(loads))
            gaps.append(gap)
            poses.append({k: row[k] for k in ("hold_time_s","base_pose","com","hand_in_target_rung")})
    return extrema(gaps, counts, times, peak), dict(reasons), late_drift(poses, cfg.hold_seconds)


def compare_trace(actual, reported):
    for key, value in actual.items():
        if value is None:
            assert reported[key] is None
        else:
            np.testing.assert_allclose(value, reported[key], rtol=0., atol=TOLERANCE)


def verify_run(path, report):
    meta, summary = read(path / "resolved.json"), read(path / "summary.json")
    model = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
    geometry = RightKneeGeometry(model)
    rows = read_knee_rows(path, model.nq)
    native = verify_native(path, rows, geometry, meta["config"]["physics_dt"])
    control, reasons, drift = verify_controls(path, rows, geometry, meta)
    claimed = summary["physics_audit"]["knee_rung3"]
    assert set(claimed["geoms"]) == geometry.names and claimed["rung"] == RUNG
    compare_trace(native, claimed["native"])
    compare_trace(control, claimed["control"])
    assert drift == summary["drift_assessment"]
    return {**verify_reaction(path, report), "knee_rung3": {"native": native, "control": control},
            "anatomy": verify_anatomy(path, meta, model), "control_failures": reasons, "drift": drift}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/g1_o6/g6_knee_clearance"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    verify(args.root, args.run_id, verify_function=verify_run)
