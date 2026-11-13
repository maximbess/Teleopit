"""Progression is recomputed from immutable run evidence, never a skip flag."""

import hashlib
import json
from pathlib import Path

from teleopit.sim.o6_diagnostics.io import read_rows

from .capacity import measured_references
from .capacity_matrix import trial_passes
from .guards import wrench_covered
from .long_hold import long_qualified


def identity(metadata):
    config = comparable_config(metadata["config"])
    value = {
        k: metadata[k]
        for k in ("grasp_pose", "controller", "support_targets", "assets")
    }
    value["config"] = config
    for key in ("ladder", "material", "physical_identity"):
        if key in metadata:
            value[key] = metadata[key]
    value["assets"] = portable_assets(value["assets"])
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def comparable_config(config):
    return {k: v for k, v in config.items() if k not in ("hold_seconds", "physics_dt")}


def portable_assets(value):
    if isinstance(value, dict):
        return {portable_assets(k): portable_assets(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_assets(v) for v in value]
    if isinstance(value, str):
        path = value.replace("\\", "/")
        for anchor in ("/teleopit/sim/", "/assets/"):
            if anchor in path and not path.startswith("https://"):
                return path[path.index(anchor) + 1 :]
    return value


def read_run(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    for name in ("summary.json", "resolved.json", "steps.jsonl"):
        digest = hashlib.sha256((path / name).read_bytes()).hexdigest()
        if manifest["files_sha256"].get(name) != digest:
            raise ValueError(f"run integrity mismatch: {path / name}")
    return {
        "path": str(path),
        "metadata": json.loads((path / "resolved.json").read_text()),
        "summary": json.loads((path / "summary.json").read_text()),
    }


def long_hold_gate(runs):
    passing = {
        r["path"]: r
        for r in runs
        if long_qualified(r["summary"], r["metadata"]["experiment"]["requested_hold_s"])
    }
    timesteps = [r["metadata"]["config"]["physics_dt"] for r in passing.values()]
    return timesteps.count(0.001) >= 2 and timesteps.count(0.0005) >= 1


def capacity_wrenches(runs, source):
    passed = {side: [] for side in ("left", "right")}
    for run in runs:
        meta, result = run["metadata"], run["summary"]
        if (
            capacity_matches(meta, source)
            and trial_passes(result)
            and not meta["fixed_base"]
            and meta["gravity_world_m_s2"] == [0, 0, -9.81]
        ):
            passed[meta["side"]].append(result["requested_wrench_world_at_mount"])
    return passed


def capacity_matches(meta, source):
    pairs = (
        ("source_grasp_pose", "grasp_pose"),
        ("controller", "controller"),
        ("source_support_targets", "support_targets"),
    )
    return (
        all(meta[a] == source[b] for a, b in pairs)
        and portable_assets(meta["assets"]) == portable_assets(source["assets"])
        and comparable_config(meta.get("source_config", {}))
        == comparable_config(source["config"])
    )


def demands_covered(run, passed):
    rows = read_rows(Path(run["path"]) / "steps.jsonl")
    demands = {s: measured_references(rows, s) for s in ("left", "right")}
    return all(
        wrench_covered(r["wrench_world_at_mount"], passed[s])
        for s, references in demands.items()
        for r in references
    )


class Evidence:
    def __init__(self, book):
        path = Path(book).resolve()
        value = json.loads(path.read_text())
        self.source = read_run(path.parent / value["source_run"])
        self.identity = identity(self.source["metadata"])
        self.runs = {
            group: [read_run(path.parent / p) for p in value.get(group, [])]
            for group in ("long_holds", "capacity_runs", "transfers")
        }
        self.runs["long_holds"] = [
            r
            for r in self.runs["long_holds"]
            if identity(r["metadata"]) == self.identity
        ]
        self.runs["transfers"] = [
            r
            for r in self.runs["transfers"]
            if identity(r["metadata"]) == self.identity
        ]
        self.passed_wrenches = capacity_wrenches(
            self.runs["capacity_runs"], self.source["metadata"]
        )

    def qualified_transfers(self):
        return [
            r
            for r in self.runs["transfers"]
            if r["summary"].get("physical_status") == "PASS"
            and demands_covered(r, self.passed_wrenches)
        ]

    def reasons(self, request):
        reasons = []
        if not long_hold_gate(self.runs["long_holds"]):
            reasons.append("stage_1_requires_120s_nominal_repeat_fine_and_convergence")
        if not demands_covered(self.source, self.passed_wrenches):
            reasons.append("stage_2_requires_both_hand_tested_wrench_reserve")
        reasons.extend(transition_prerequisites(request, self.qualified_transfers()))
        return reasons


def transition_prerequisites(request, qualified):
    experiments = [r["metadata"]["experiment"] for r in qualified]
    target, mode = request["target_share"], request["mode"]

    def has(kind, **criteria):
        return any(
            e["mode"] == kind
            and e["target_share"] == target
            and all(e.get(k) == v for k, v in criteria.items())
            for e in experiments
        )

    checks = {}
    if mode == "load":
        checks = load_prerequisites(request, experiments, has)
    else:
        checks = hand_prerequisites(request, has)
    return [key for key, ok in checks.items() if not ok]


def load_prerequisites(request, experiments, has):
    checks = {}
    previous = {0.1: 0.05, 0.2: 0.1}.get(request["target_share"])
    if previous:
        checks["previous_load_level"] = any(
            e["mode"] == "load" and e["target_share"] == previous for e in experiments
        )
    if request["plateau_seconds"] == 30:
        checks["ten_second_load_plateau"] = has("load", plateau_seconds=10)
    return checks


def hand_prerequisites(request, has):
    mode, side = request["mode"], request["side"]
    checks = {}
    checks["qualified_stage_3_target"] = has("load")
    if mode in ("release", "reacquire"):
        checks["both_sides_closed_unload_reload"] = all(
            has("unload", side=s) for s in ("left", "right")
        )
    if mode == "release" and request["three_seconds"] > 1:
        checks["preceding_three_support_duration"] = has(
            "release", side=side, three_seconds={3: 1, 10: 3}[request["three_seconds"]]
        )
    if mode == "reacquire":
        checks.update(cycle_prerequisites(request, has))
    return checks


def cycle_prerequisites(request, has):
    checks = {
        "both_sides_ten_second_three_support": all(
            has("release", side=s, three_seconds=10) for s in ("left", "right")
        )
    }
    if request["cycles"] > 1:
        checks["both_sides_one_cycle"] = all(
            has("reacquire", side=s, cycles=1) for s in ("left", "right")
        )
    if request["alternating"]:
        checks["both_sides_three_cycles"] = all(
            has("reacquire", side=s, cycles=3, alternating=False)
            for s in ("left", "right")
        )
    return checks
