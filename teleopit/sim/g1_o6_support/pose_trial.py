"""Pose candidates use the existing environment and the same strict HOLD runner."""

import json
from pathlib import Path

import mujoco
import numpy as np

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.diagnostics import snapshot
from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.grasp_pose import mount_target
from teleopit.sim.g1_o6.initialize import arm_joints
from teleopit.sim.g1_o6.posture_report import annotate
from teleopit.sim.o6_diagnostics.io import manifest, output_directory, write_json

from .long_hold import experiment_metadata, run_hold
from .pose_controller import FingerExperiment
from .pose_observer import PoseObserver


class SearchEnvironment(Environment):
    def __init__(self, candidate, seconds, dt):
        self.candidate = candidate
        self.reset_arm_seed = candidate.ik_seed
        super().__init__(
            Config(hold_seconds=seconds, physics_dt=dt, **candidate.reset_offsets),
            pose=candidate.pose,
        )

    def reset(self):
        row = super().reset()
        self.hold_controller = FingerExperiment(self, self.candidate.controller)
        return row


def rung_transforms(env):
    return {
        str(r): {
            "name": f"left_ladder_rung_{r:02d}",
            "position_world_m": env.data.geom(
                f"left_ladder_rung_{r:02d}"
            ).xpos.tolist(),
            "rotation_world": env.data.geom(f"left_ladder_rung_{r:02d}").xmat.tolist(),
            "half_sizes_m": env.model.geom(f"left_ladder_rung_{r:02d}").size.tolist(),
        }
        for r in (4, 10, 11)
    }


def reset_report(env):
    model, data = env.model, env.data
    arms = {
        s: {
            model.joint(j).name: float(data.qpos[model.jnt_qposadr[j]])
            for j in arm_joints(model, s)
        }
        for s in env.hands
    }
    targets = {
        s: {"position_world_m": p.tolist(), "rotation_world": r.tolist()}
        for s, rung in zip(env.hands, env.pose.hand_rungs)
        for p, r in [mount_target(env, env.pose, s, rung)]
    }
    transforms = rung_transforms(env)
    mounts = [data.site(h["mount"]).xpos.copy() for h in env.hands.values()]
    q = data.qpos[model.jnt_qposadr[1:]]
    margin = np.minimum(q - model.jnt_range[1:, 0], model.jnt_range[1:, 1] - q)
    return {
        "ik": env.ik,
        "arm_joints_rad": arms,
        "mount_targets": targets,
        "rung_transforms": transforms,
        "rung_11_minus_10_world_m": (
            np.array(transforms["11"]["position_world_m"])
            - transforms["10"]["position_world_m"]
        ).tolist(),
        "actual_mount_separation_m": float(np.linalg.norm(mounts[0] - mounts[1])),
        "minimum_joint_margin_rad": float(margin.min()),
        "spacing_semantics": "signed lateral offset per hand, not total separation",
        "offset_semantics": "world XYZ; tilt also rotates the reference mount translation",
    }


def candidate_metadata(env):
    return {
        "candidate": {
            **env.candidate.record(),
            "configuration_sha256": env.candidate.digest(),
        },
        "controller": env.hold_controller.metadata(env),
        "controller_initial_state": env.hold_controller.state(),
        "reset_inspection": reset_report(env),
        "pose_diagnostic_schema": "g1_o6_pose_diagnostic_v1",
        "contact_loaded_threshold_N": 0.05,
        "friction_usage": "translation-only elliptical tangential ratio; torsional ratio separate, neither is a full contact-cone margin",
        "power_method": "qfrc_actuator[dof] * qvel[dof]; physics-step rectangular work integration",
        "wrist_wrench": "parent-on-hand; world axes, moment at current wrist mount; internal robot wrench",
        "physics_windows": {
            "pre_s": 0.25,
            "post_s": 0.25,
            "native_solver_data_max_age_steps": 1,
        },
        "clearance_distance_cap_m": 0.1,
    }


def reject_reset(env, path, extra, reasons, status="INFEASIBLE"):
    output_directory(path)
    initial = annotate(env, snapshot(env))
    metadata = experiment_metadata(
        env,
        initial,
        {"mode": "pose-search", "requested_hold_s": env.config.hold_seconds},
    )
    metadata.update(extra)
    write_json(path / "resolved.json", metadata)
    mujoco.mj_saveLastXML(str(path / "scene.xml"), env.model)
    result = {
        "schema": "g1_o6_pose_rejection_v1",
        "software_status": "PASS",
        "reset_status": status,
        "strict_status": "NOT RUN",
        "reasons": reasons,
        "physics_steps": 0,
        "progression_justified": False,
    }
    write_json(path / "summary.json", result)
    manifest(path, "pose_reset_rejection", {})
    return result


def run_candidate(candidate, output, *, seconds=30, physics_dt=0.001):
    path = Path(output)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(path)
    env, extra, reasons = prepare_candidate(candidate, seconds, physics_dt)
    if reasons:
        return reject_reset(env, path, extra, reasons)
    return run_hold(
        path,
        seconds=seconds,
        physics_dt=physics_dt,
        environment=env,
        observer_factory=PoseObserver,
        metadata_extra=extra,
    )


def prepare_candidate(candidate, seconds, dt):
    env = SearchEnvironment(candidate, seconds, dt)
    extra = candidate_metadata(env)
    reasons = list(env.initial_errors)
    if extra["reset_inspection"]["minimum_joint_margin_rad"] < -1e-9:
        reasons.append("initial_joint_range")
    return env, extra, reasons


def duplicate_reset(qpos, finger_identity, references):
    for reference in references:
        if (
            reference["finger_identity"] == finger_identity
            and np.max(np.abs(qpos - reference["qpos"])) <= 0.001
        ):
            return reference["id"]
    return None


def preflight_geometry(candidate, output, references):
    env, extra, reasons = prepare_candidate(candidate, 30, 0.001)
    if reasons:
        return reject_reset(env, output, extra, reasons)
    identity = json.dumps(extra["controller"], sort_keys=True)
    duplicate = duplicate_reset(env.data.qpos, identity, references)
    if duplicate is not None:
        return reject_reset(
            env, output, extra, ["near_identical_reset_to:" + duplicate], "DUPLICATE"
        )
    references.append(
        {
            "id": candidate.id,
            "qpos": env.data.qpos.tolist(),
            "finger_identity": identity,
        }
    )
    return None


def prior_resets(root, records):
    references = []
    for record in records:
        if record["stage"] != "screen" or record["status"] != "COMPLETE":
            continue
        if record["result"].get("reset_status") in ("INFEASIBLE", "DUPLICATE"):
            continue
        meta = json.loads((root / record["output"] / "resolved.json").read_text())
        if meta["candidate"]["kind"] == "geometry":
            references.append(
                {
                    "id": record["candidate"],
                    "qpos": meta["initial"]["base_pose"]
                    + meta["initial"]["joint_positions"],
                    "finger_identity": json.dumps(meta["controller"], sort_keys=True),
                }
            )
    return references
