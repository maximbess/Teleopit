"""Bounded ladder/material campaign using the existing single-writer ledger."""

import hashlib
import json
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.grasp_pose import GraspPose
from teleopit.sim.g1_o6.material import Material
from teleopit.sim.g1_o6.metadata import controller_settings
from teleopit.sim.g1_o6.model import provenance as asset_provenance
from teleopit.sim.ladder_config import compiled_ladder, geometry_matrix
from teleopit.sim.o6_diagnostics.io import write_json

from .ladder_identity import assert_preserved, physical_identity
from .ladder_observer import LadderObserver
from .ladder_reset import exact_pose_inspections, reset_candidates, select_reset
from .long_hold import long_qualified, run_hold
from .pose_campaign import STAGES, Campaign
from .pose_results import motion_cost
from .pose_trial import reject_reset, reset_report

REFERENCE = "cc0cd5393c670e7801bb7d242c700e6af4d47303"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def conditions():
    rows = [
        {"id": g + "_reference", "geometry": g, "material": None, "stage_group": "A"}
        for g in geometry_matrix()
    ]
    for group, scope, geometries, coefficients in (
        ("B", "H", ("G0", "G5", "G6"), (0.4, 0.6, 0.8, 1.2)),
        ("C", "M", ("G0", "G6"), (0.6, 0.8, 1.2)),
    ):
        rows.extend(
            {
                "id": f"{g}_{scope}_{mu}".replace(".", "p"),
                "geometry": g,
                "material": {"scope": scope, "mu": mu},
                "stage_group": group,
            }
            for g in geometries
            for mu in coefficients
        )
    return rows


def specification():
    return {
        "schema": "g1_o6_ladder_material_campaign_v1",
        "reference_revision": REFERENCE,
        "geometries": {k: asdict(v) for k, v in geometry_matrix().items()},
        "conditions": conditions(),
        "reset_shortlists": {
            k: reset_candidates(v) for k, v in geometry_matrix().items()
        },
        "budgets": dict(screen=25, **{"60s": 8, "120s": 6, "repeat": 2, "fine": 2}),
        "controller": controller_settings(GraspPose(), Config()),
        "reset_rule": "first feasible candidate in frozen geometric order; exact pose inspected separately",
        "ranking": "strict duration and anatomical contact integrity; late-motion cost; peak actuator fraction; identity",
        "reservations": ["G5", "G6", "lower friction H", "G6 whole-tread M"],
        "long_diagnostic": "preceding strict/contact pass required; nonconvergent shorter runs may diagnose long behavior but are not stable",
        "matched_windows": {"screen": [20, 30], "60s": [50, 60], "120s": [90, 120]},
        "qualification": "120s nominal + fresh repeat + .0005 s physics; strict, anatomical, no-assistance, prohibited-support and late-motion gates all pass",
        "aliases": "H mu=1.8 may reuse native geometry reference only after selected-pair equivalence verification",
    }


def code_provenance():
    import mujoco
    import numpy
    import scipy

    files = sorted(Path("teleopit/sim").rglob("*.py"))
    return {
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty_diff": subprocess.check_output(["git", "diff", "HEAD"], text=True),
        "source_sha256": {
            p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "assets": asset_provenance(),
        "controller": controller_settings(GraspPose(), Config()),
        "seed": 0,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco": mujoco.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
    }


def run_identity(*, condition, reset, seconds, dt, provenance):
    return digest(
        {
            "condition": condition,
            "reset": reset,
            "seconds": seconds,
            "physics_dt": dt,
            "control_dt": 0.02,
            "provenance": provenance,
        }
    )


def screen_passes(result):
    audit = result.get("physics_audit", {})
    return (
        result.get("strict_status") == "PASS"
        and result.get("contact_quality", {}).get("status") == "PASS"
        and audit.get("no_external_support") is True
        and audit.get("unintended_support_steps") == 0
        and audit.get("housing_contact_steps") == 0
    )


def condition_status(records):
    if not records:
        return "NOT RUN"
    if records[0]["result"].get("reset_status") == "INFEASIBLE":
        return "INFEASIBLE"
    stages = {r["stage"]: r["result"] for r in records}
    if all(
        long_qualified(stages.get(s, empty_result()), 120)
        for s in ("120s", "repeat", "fine")
    ):
        return "QUALIFIED"
    if stages.get("120s", {}).get("strict_status") == "PASS":
        return "duration passed, stability not qualified"
    return (
        "FAILED"
        if any(r.get("strict_status") == "FAIL" for r in stages.values())
        else "NOT RUN (long qualification)"
    )


def empty_result():
    return {
        "physics_audit": {},
        "strict_status": "NOT RUN",
        "drift_assessment": {"status": "NOT RUN"},
    }


def trial_payload(condition, reset):
    return {
        "condition": condition,
        "reset": reset,
        "ladder": asdict(geometry_matrix()[condition["geometry"]]),
    }


def run_condition(payload, output, seconds, dt):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    condition, reset = payload["condition"], payload["reset"]
    ladder = geometry_matrix()[condition["geometry"]]
    material = Material(**condition["material"]) if condition["material"] else None
    if reset["status"] == "INFEASIBLE":
        result = {
            "software_status": "PASS",
            "reset_status": "INFEASIBLE",
            "strict_status": "NOT RUN",
            "reasons": ["no feasible declared reset"],
            "progression_justified": False,
        }
        output.mkdir(parents=True)
        write_json(output / "summary.json", result)
        write_json(output / "resolved.json", payload)
        return result
    env = Environment(
        Config(hold_seconds=seconds, physics_dt=dt),
        ladder=ladder,
        material=material,
        pose=GraspPose(**reset["pose"]),
    )
    physical = physical_identity(env.model)
    assert_preserved(physical, payload["reference_physics"])
    np.testing.assert_array_equal(
        env.data.qpos,
        reset["inspections"][-1]["qpos"],
        err_msg="frozen reset changed before material trial",
    )
    extra = {
        "ladder": compiled_ladder(env.model, ladder),
        "material": env.audit.get("material"),
        "condition": condition,
        "frozen_reset": reset,
        "exact_reference_inspection": payload.get("exact_reference_inspection"),
        "reset_inspection": reset_report(env),
        "physical_identity": physical,
        "exact_command": payload.get("command"),
        "run_identity": payload.get("run_identity"),
    }
    if env.initial_errors:
        return reject_reset(env, output, extra, env.initial_errors)
    return run_hold(
        output,
        seconds=seconds,
        physics_dt=dt,
        environment=env,
        observer_factory=LadderObserver,
        metadata_extra=extra,
    )


class LadderCampaign(Campaign):
    """Reuse atomic persistence and frozen decisions; isolate the new budget."""

    def __init__(self, root, workers=3):
        self.root, self.workers = Path(root), workers
        if not 1 <= workers <= 6:
            raise ValueError("workers must be in 1..6")
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "campaign_state.json"
        spec = specification()
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["spec_sha256"] != digest(spec):
                raise ValueError("frozen campaign specification changed")
        else:
            self.state = {
                "schema": spec["schema"],
                "specification": spec,
                "spec_sha256": digest(spec),
                "runs": [],
                "decisions": [],
                "candidates": {r["id"]: r for r in conditions()},
            }
            self.save()

    def candidate(self, name):
        condition = self.state["candidates"][name]
        return SimpleNamespace(digest=lambda: digest(condition))

    def prepare(self):
        self.decision(
            "reference_physics", lambda: physical_identity(Environment().model)
        )
        for identity, geometry in geometry_matrix().items():
            self.decision("reset_" + identity, lambda g=geometry: select_reset(g))
        self.decision("reference_material_equivalence", reference_material_equivalence)
        self.decision("exact_reference_qpos_inspections", exact_pose_inspections)
        write_json(self.root / "frozen_campaign.json", self.state)

    def reset_for(self, condition):
        name = "reset_" + condition["geometry"]
        return next(r["value"] for r in self.state["decisions"] if r["name"] == name)

    def validate_stage(self, name, stage, records=None):
        records = self.state["runs"] if records is None else records
        limits = self.state["specification"]["budgets"]
        if stage not in limits:
            raise ValueError("unknown stage")
        if (
            len({r["candidate"] for r in records if r["stage"] == stage})
            >= limits[stage]
        ):
            raise ValueError("campaign trial budget exhausted")
        self.validate_preceding(name, stage)

    def validate_preceding(self, name, stage):
        preceding = {
            "60s": "screen",
            "120s": "60s",
            "repeat": "120s",
            "fine": "repeat",
        }.get(stage)
        if preceding:
            prior = self.previous(name, preceding)
            if prior is None or not screen_passes(prior.get("result", {})):
                raise ValueError("preceding strict/contact/physical gates failed")
            if stage in ("repeat", "fine") and not long_qualified(prior["result"], 120):
                raise ValueError("preceding long trial is not qualified")

    def run_many(self, requests, workers=None):
        plans = self.plan_requests(requests)
        pending = []
        for name, stage, identity, payload in plans:
            record = self.reserve(name, stage, None, None)
            record.update(identity=identity, payload=payload)
            self.save()
            pending.append(record)
        if pending:
            self.execute(pending, workers or self.workers)

    def plan_requests(self, requests):
        if len(set(requests)) != len(requests):
            raise ValueError("duplicate request in batch")
        provenance = code_provenance()
        pending = []
        preview = list(self.state["runs"])
        for name, stage in requests:
            condition = self.state["candidates"][name]
            reset = self.reset_for(condition)
            seconds, dt = STAGES[stage]
            identity = run_identity(
                condition=condition,
                reset=reset,
                seconds=seconds,
                dt=dt,
                provenance=provenance,
            )
            prior = self.previous(name, stage)
            if prior:
                self.verify_resume(prior, identity)
                continue
            self.validate_stage(name, stage, preview)
            payload = trial_payload(condition, reset)
            payload["exact_reference_inspection"] = next(
                d["value"]
                for d in self.state["decisions"]
                if d["name"] == "exact_reference_qpos_inspections"
            )[condition["geometry"]]
            payload["reference_physics"] = next(
                d["value"]
                for d in self.state["decisions"]
                if d["name"] == "reference_physics"
            )
            payload.update(
                run_identity=identity,
                provenance=provenance,
                command=f'python -m teleopit.sim.g1_o6_support.ladder_experiment --output "{self.root.as_posix()}" trial {name} {stage}',
            )
            pending.append((name, stage, identity, payload))
            preview.append({"candidate": name, "stage": stage})
        return pending

    def retry(self, name, stage, reason):
        prior = self.previous(name, stage)
        if (
            not reason.strip()
            or prior is None
            or prior["status"] not in ("ERROR", "RUNNING")
        ):
            raise ValueError(
                "technical retry requires an interrupted/error attempt and a reason"
            )
        condition = self.state["candidates"][name]
        seconds, dt = STAGES[stage]
        identity = run_identity(
            condition=condition,
            reset=self.reset_for(condition),
            seconds=seconds,
            dt=dt,
            provenance=code_provenance(),
        )
        if identity != prior["identity"]:
            raise ValueError(
                "technical retry identity changed; start a separately reviewed campaign"
            )
        record = self.reserve(name, stage, prior, reason)
        record.update(identity=identity, payload=prior["payload"])
        self.save()
        self.execute([record], 1)

    def verify_resume(self, record, identity):
        if record["status"] != "COMPLETE":
            raise ValueError(
                "interrupted/error run retained; explicit technical recovery required"
            )
        if record.get("identity") != identity:
            raise ValueError(
                "resume identity differs in code, assets, reset or condition"
            )
        for name, checksum in record["files_sha256"].items():
            if file_hash(self.root / record["output"] / name) != checksum:
                raise ValueError("immutable run evidence checksum mismatch")

    def execute(self, records, workers):
        errors = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = {
                pool.submit(
                    run_condition,
                    r["payload"],
                    self.root / r["output"],
                    *STAGES[r["stage"]],
                ): r
                for r in records
            }
            for future in as_completed(jobs):
                record = jobs[future]
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001 - preserve worker error, then fail the batch
                    record.update(status="ERROR", error=repr(error))
                    errors.append(record["id"])
                    self.save()
                else:
                    record["files_sha256"] = run_checksums(self.root / record["output"])
                    self.complete(record, result)
        if errors:
            raise RuntimeError("technical failures preserved: " + ", ".join(errors))


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run_checksums(path):
    return {
        p.relative_to(path).as_posix(): file_hash(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def reference_material_equivalence():
    from teleopit.sim.g1_o6.model import build_model

    reports = {}
    for identity in ("G0", "G5", "G6"):
        geometry = geometry_matrix()[identity]
        original = build_model(Config(), ladder=geometry)[0]
        equivalent, _, _, audit = build_model(
            Config(), ladder=geometry, material=Material("H", 1.8)
        )
        fields = []
        for name in dir(original):
            if isinstance(getattr(original, name), np.ndarray):
                np.testing.assert_array_equal(
                    getattr(original, name), getattr(equivalent, name), err_msg=name
                )
                fields.append(name)
        reports[identity] = {
            "status": "PASS",
            "identical_compiled_fields": len(fields),
            "eligible_native_pairs": len(audit["material"]["pairs"]),
        }
    return reports


def ranked_promotions(campaign, preceding, maximum):
    records = promotion_records(campaign, preceding)
    records.sort(key=promotion_rank)
    selected = reserved_promotions(records, campaign.state["candidates"])
    selected.extend(r["candidate"] for r in records if r["candidate"] not in selected)
    return {
        "ranked": [r["candidate"] for r in records],
        "selected": selected[:maximum],
        "diagnostic_nonconvergent": [
            r["candidate"]
            for r in records
            if r["result"]["drift_assessment"]["status"] != "PASS"
        ],
    }


def promotion_records(campaign, preceding):
    return [
        r
        for r in campaign.state["runs"]
        if r["stage"] == preceding
        and r["status"] == "COMPLETE"
        and screen_passes(r["result"])
    ]


def promotion_rank(record):
    cost = motion_cost(record["result"])
    return (
        cost if cost is not None else float("inf"),
        record["result"]["strict"]["peak_effort_fraction"],
        record["candidate"],
    )


def reservation_groups(condition):
    geometry, material = condition["geometry"], condition["material"]
    groups = [geometry]
    if material:
        groups.append(material["scope"])
        groups.append(geometry + "_" + material["scope"])
    return groups


def reserved_promotions(records, candidates):
    selected = []
    for group in ("G5", "G6", "H", "G6_M"):
        match = next(
            (
                r["candidate"]
                for r in records
                if group in reservation_groups(candidates[r["candidate"]])
                and r["candidate"] not in selected
            ),
            None,
        )
        if match:
            selected.append(match)
    return selected
