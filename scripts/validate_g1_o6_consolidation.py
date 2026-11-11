"""Compare fresh current holds with explicit-preset references on the same runtime.

Reference layout and zero tolerances are documented in docs/g1_o6/consolidation.
Output must be new: this command never replaces an earlier experiment.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.run import run_trial


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_arrays(config, archive):
    env = Environment(config)
    with np.load(archive) as arrays:
        actual = {
            name: getattr(env.model, name)
            for name in arrays.files
            if not name.startswith("initial_")
        }
        actual.update(initial_qpos=env.data.qpos, initial_ctrl=env.data.ctrl)
        changed = [
            name
            for name in arrays.files
            if not np.array_equal(arrays[name], actual[name])
        ]
        return {"count": len(arrays.files), "changed": changed}


class HoldAudit:
    def __init__(self):
        self.steps = 0
        self.nonzero_applied_steps = 0
        self.housing_contact_steps = 0
        self.housing = None
        self.ladder = None

    def __call__(self, env):
        if env.phase != "HOLD":
            return
        self.steps += 1
        self.nonzero_applied_steps += int(
            np.any(env.data.qfrc_applied) or np.any(env.data.xfrc_applied)
        )
        if self.housing is None:
            self.select_geoms(env.model)
        self.housing_contact_steps += int(
            any(
                (c.geom1 in self.housing and c.geom2 in self.ladder)
                or (c.geom2 in self.housing and c.geom1 in self.ladder)
                for c in env.data.contact
            )
        )

    def select_geoms(self, model):
        names = [model.body(i).name for i in model.geom_bodyid]
        self.housing = {
            i
            for i, name in enumerate(names)
            if "wrist_roll" in name or "wrist_pitch" in name
        }
        self.ladder = {i for i in range(model.ngeom) if "ladder" in model.geom(i).name}

    def report(self):
        return {
            name: getattr(self, name)
            for name in ("steps", "nonzero_applied_steps", "housing_contact_steps")
        }


def compare_run(reference, current):
    # Every logged physical field, phase event and summary must match, not just PASS.
    files = {
        name: digest(reference / name) == digest(current / name)
        for name in ("steps.jsonl", "summary.json", "scene.xml")
    }
    before, after = [read(path / "resolved.json") for path in (reference, current)]
    metadata = {
        name: before[name] == after[name]
        for name in (
            "assets",
            "model",
            "ik",
            "initial",
            "support_targets",
            "grasp_pose",
        )
    }
    metadata["retained_config"] = all(
        before["config"][k] == v for k, v in after["config"].items()
    )
    traces = {
        s: read(reference / f"{s}.wrist.json")["samples"]
        == read(current / f"{s}.wrist.json")["samples"]
        for s in ("left", "right")
    }
    return {
        "files_equal": files,
        "metadata_equal": metadata,
        "wrist_samples_equal": traces,
    }


def qualification(reference, output, config):
    audit = HoldAudit()
    summary = run_trial(config, output, physics_observer=audit)
    comparison = compare_run(reference, output)
    rows = [
        json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()
    ]
    comparison.update(
        hold_physics=audit.report(),
        final_phase=summary["final_phase"],
        hold_seconds=summary["unsupported_hold_seconds"],
        max_hold_drift_m=max(r["base_drift_m"] for r in rows if r["phase"] == "HOLD"),
        max_hold_wrist_bend_deg=summary["posture"]["max_hold_wrist_bend_deg"],
        min_hold_housing_clearance_m=summary["posture"]["min_hold_housing_clearance_m"],
        peak_effort_fraction=summary["peak_effort_fraction"],
    )
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [
        ("nominal", "before_10s", Config()),
        ("repeat", "before_10s", Config()),
        ("long", "before_30s", Config(hold_seconds=30)),
        ("fine", "before_fine_30s", Config(physics_dt=0.0005, hold_seconds=30)),
    ]
    report = {
        "tolerances": read(args.reference_root / "tolerances.json"),
        "arrays": {},
        "runs": {},
    }
    for label, config in (("default", Config()), ("fine", cases[-1][2])):
        report["arrays"][label] = compare_arrays(
            config, args.reference_root / f"before_{label}_arrays.npz"
        )
    for name, reference, config in cases:
        report["runs"][name] = qualification(
            args.reference_root / reference, args.output / name, config
        )
        (args.output / "comparison.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(name, json.dumps(report["runs"][name]), flush=True)
    assert all(not a["changed"] for a in report["arrays"].values())
    for run in report["runs"].values():
        assert all(
            all(run[field].values())
            for field in ("files_equal", "metadata_equal", "wrist_samples_equal")
        )
        assert run["final_phase"] == "HOLD_COMPLETE"
        assert run["hold_physics"]["nonzero_applied_steps"] == 0
        assert run["hold_physics"]["housing_contact_steps"] == 0


if __name__ == "__main__":
    main()
