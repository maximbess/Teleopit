"""Progression is recomputed from immutable run evidence, never a skip flag."""

import hashlib
import json
from pathlib import Path

from teleopit.sim.o6_diagnostics.io import read_rows

from .capacity import measured_references, sampled_state_key
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
    for key in ("ladder", "material", "physical_identity", "environment_spec"):
        if key in metadata:
            value[key] = (
                dict(metadata[key]) if key == "physical_identity" else metadata[key]
            )
    if "physical_identity" in value:
        value["physical_identity"].pop("physics_dt", None)
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
    by_target = target_capacity_wrenches(runs, source)
    return {
        s: by_target[s].get(source["support_targets"].get(s + "_hand"), [])
        for s in ("left", "right")
    }


def target_capacity_wrenches(runs, source):
    passed = {side: {} for side in ("left", "right")}
    for run in runs:
        meta, result = run["metadata"], run["summary"]
        if (
            capacity_matches(meta, source)
            and trial_passes(result)
            and not meta["fixed_base"]
            and meta["gravity_world_m_s2"] == [0, 0, -9.81]
        ):
            side = meta["side"]
            target = meta.get(
                "source_sample_support_targets", meta["source_support_targets"]
            ).get(side + "_hand")
            passed[side].setdefault(target, []).append(
                result["requested_wrench_world_at_mount"]
            )
    return passed


def capacity_matches(meta, source):
    pairs = (
        ("source_grasp_pose", "grasp_pose"),
        ("controller", "controller"),
        ("source_support_targets", "support_targets"),
    )
    return (
        all(meta.get(a) == source.get(b) for a, b in pairs)
        and capacity_scene_matches(meta, source)
        and portable_assets(meta["assets"]) == portable_assets(source["assets"])
        and comparable_config(meta.get("source_config", {}))
        == comparable_config(source["config"])
    )


def capacity_scene_matches(meta, source):
    keys = ("ladder", "material", "physical_identity", "environment_spec", "initial")
    return all(
        "source_" + key in meta
        and scene_field(meta["source_" + key], key) == scene_field(source[key], key)
        for key in keys
        if key in source
    )


def scene_field(value, key):
    return (
        {k: v for k, v in value.items() if k != "physics_dt"}
        if key == "physical_identity"
        else value
    )


def demands_covered(run, passed, by_target=None, capacity_runs=None):
    rows = read_rows(Path(run["path"]) / "steps.jsonl")
    if by_target is not None:
        meta = run.get("metadata") or json.loads(
            (Path(run["path"]) / "resolved.json").read_text()
        )
        return targeted_demands_covered(
            rows, by_target, meta["support_targets"], capacity_runs=capacity_runs
        )
    demands = {s: measured_references(rows, s) for s in ("left", "right")}
    return all(
        wrench_covered(r["wrench_world_at_mount"], passed[s])
        for s, references in demands.items()
        for r in references
    )


def critical_groups(rows, initial_targets):
    from .g6_gates import MOVING_PHASES

    groups = {}
    for row in rows:
        if row["phase"] != "HOLD" or row.get("assistance_gain", 1) != 0:
            continue
        phase = row.get("experiment_phase", row["phase"])
        targets = row.get("g6", {}).get(
            "support_targets", row.get("support_targets", initial_targets)
        )
        selected = row.get("transfer", {}).get("side")
        for side in ("left", "right"):
            if phase in MOVING_PHASES and selected == side:
                continue
            groups.setdefault((side, targets[side + "_hand"], phase), []).append(row)
    return groups


def targeted_demands_covered(rows, by_target, initial_targets, *, capacity_runs=None):
    groups = critical_groups(rows, initial_targets)
    demands = [
        (side, target, sample)
        for (side, target, _phase), samples in groups.items()
        for sample in measured_references(samples, side)
    ]
    return bool(demands) and all(
        wrench_covered(
            sample["wrench_world_at_mount"],
            state_wrenches(rows, side, target, sample, initial_targets, capacity_runs)
            if capacity_runs is not None
            else by_target[side].get(target, []),
        )
        for side, target, sample in demands
    )


def state_wrenches(rows, side, target, sample, initial_targets, capacity_runs):
    row = next(r for r in rows if r["time"] == sample["time_s"])
    targets = row.get("g6", {}).get(
        "support_targets", row.get("support_targets", initial_targets)
    )
    key = sampled_state_key(row, targets)
    return [
        r["summary"]["requested_wrench_world_at_mount"]
        for r in capacity_runs
        if r["metadata"]["side"] == side
        and r["metadata"].get("source_sample", {}).get("state_sha256") == key
        and r["metadata"].get("source_sample_support_targets", {}).get(side + "_hand")
        == target
        and trial_passes(r["summary"])
    ]


class Evidence:
    def __init__(self, book):
        path = Path(book).resolve()
        value = json.loads(path.read_text())
        self.source = read_run(path.parent / value["source_run"])
        self.identity = identity(self.source["metadata"])
        self.runs = load_evidence_runs(path, value, self.identity)
        self.passed_wrenches = capacity_wrenches(
            self.runs["capacity_runs"], self.source["metadata"]
        )
        self.target_wrenches = target_capacity_wrenches(
            self.runs["capacity_runs"], self.source["metadata"]
        )
        self.is_g6 = (
            self.source["metadata"]
            .get("ladder", {})
            .get("configuration", {})
            .get("identity")
            == "G6"
        )
        if value.get("campaign") == "g6_stability" and not self.is_g6:
            raise ValueError("G6 evidence requires resolved G6 identity")
        if self.is_g6:
            self.runs["capacity_runs"] = matching_capacity_runs(
                self.runs["capacity_runs"], self.source["metadata"]
            )

    def covered(self, run):
        return demands_covered(
            run,
            self.passed_wrenches,
            self.target_wrenches if self.is_g6 else None,
            self.runs["capacity_runs"] if self.is_g6 else None,
        )

    def qualified_transfers(self):
        return [
            {**r, "summary": {**r["summary"], "capacity_status": "PASS"}}
            for r in self.runs["transfers"]
            if r["summary"].get("physical_status") == "PASS" and self.covered(r)
        ]

    def reasons(self, request):
        from .g6_gates import initial_pose_gate

        reasons = []
        gate = initial_pose_gate if self.is_g6 else long_hold_gate
        if not gate(self.runs["long_holds"]):
            reasons.append("stage_1_requires_120s_nominal_repeat_fine_and_convergence")
        if not self.covered(self.source):
            reasons.append("stage_2_requires_both_hand_tested_wrench_reserve")
        qualified = self.qualified_transfers()
        reasons.extend(transition_prerequisites(request, qualified))
        return reasons + self.g6_reasons(request, qualified)

    def g6_reasons(self, request, qualified):
        from .g6_gates import loaded_hold_gate

        reasons = []
        if request["mode"].startswith("adjacent") and not self.has_higher_tread():
            reasons.append("adjacent_higher_G6_tread_required")
        if self.is_g6 and request["mode"] != "load" and not loaded_hold_gate(qualified):
            reasons.append("g6_requires_30_percent_sixty_second_repeat_and_fine")
        return reasons

    def has_higher_tread(self):
        hands = [
            t
            for n, t in self.source["metadata"]["support_targets"].items()
            if n.endswith("hand")
        ]
        return self.is_g6 and all(int(t.rsplit("_", 1)[1]) < 8 for t in hands)


def load_evidence_runs(path, value, source_identity):
    runs = {
        group: [read_run(path.parent / p) for p in value.get(group, [])]
        for group in ("long_holds", "capacity_runs", "transfers")
    }
    for group in ("long_holds", "transfers"):
        runs[group] = [
            r for r in runs[group] if identity(r["metadata"]) == source_identity
        ]
    return runs


def matching_capacity_runs(runs, source):
    return [
        r
        for r in runs
        if capacity_matches(r["metadata"], source)
        and not r["metadata"]["fixed_base"]
        and r["metadata"]["gravity_world_m_s2"] == [0, 0, -9.81]
    ]


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
    previous = {0.1: 0.05, 0.2: 0.1, 0.3: 0.2}.get(request["target_share"])
    if previous:
        checks["previous_load_level"] = any(
            e["mode"] == "load" and e["target_share"] == previous for e in experiments
        )
    if request["plateau_seconds"] == 30:
        checks["ten_second_load_plateau"] = has("load", plateau_seconds=10)
    if request["plateau_seconds"] == 60:
        checks["thirty_second_load_plateau"] = has("load", plateau_seconds=30)
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
    if mode in ("adjacent-first", "adjacent"):
        checks["both_orders_same_tread_cycles"] = all(
            has("reacquire", side=s, cycles=3, alternating=True)
            for s in ("left", "right")
        )
        if mode == "adjacent":
            checks["first_destination_capacity_qualified"] = has(
                "adjacent-first", side=side
            )
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
