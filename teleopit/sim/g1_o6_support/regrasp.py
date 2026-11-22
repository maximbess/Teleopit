"""Run physical CPU G1/O6 rung-7 regrasp trials from the verified right-wrist reference."""

import argparse
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from teleopit.sim.o6_grasp.controller import smooth

from .g6_wrist_grip_experiment import WristObserver
from .ladder_campaign import file_hash
from .ladder_identity import physical_identity
from .long_hold import run_hold
from .pose_arm import rung_pose
from .regrasp_actions import PhysicalHands, fixed_model
from .regrasp_measure import STATES, SUPPORTS
from .regrasp_trial import (
    ANGLE_TOL_RAD,
    CRITERIA,
    MOVE_S,
    POSITION_TOL_M,
    Trial,
    WristPath,
)
from .scene_contract import restore_environment

SOURCE = Path("outputs/g1_o6/g6_thumb_reference/runs/control_screen")


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def verified_environment(source):
    manifest = json.loads((source / "manifest.json").read_bytes())["files_sha256"]
    for name in ("resolved.json", "scene.mjb", "steps.jsonl"):
        if file_hash(source / name) != manifest[name]:
            raise ValueError("source hash mismatch for " + name)
    meta = json.loads((source / "resolved.json").read_bytes())
    env = restore_environment(meta, 0.001, saved_scene=source / "scene.mjb")
    if meta["environment_spec"]["grasp_acquisition"].get("wrist_mode") != "right":
        raise ValueError("expected the G6 right-wrist reference")
    return env, meta, manifest


def reproduce(source, output):
    env, meta, manifest = verified_environment(source)
    extra = {
        k: meta[k] for k in ("g6_campaign", "reset_inspection", "next_tread_reach")
    }
    result = run_hold(
        output,
        seconds=30.0,
        physics_dt=0.001,
        environment=env,
        observer_factory=WristObserver,
        metadata_extra=extra,
    )
    if (
        result["strict_status"] != "PASS"
        or file_hash(output / "steps.jsonl") != manifest["steps.jsonl"]
    ):
        raise ValueError("30 s physical reference differed from its archive")


def run_branch(args, directory, matched=None):
    directory.mkdir(parents=True, exist_ok=False)
    env, meta, hashes = verified_environment(args.source)
    probe = getattr(args, "probe", None)
    prepare_scene(env, args, directory, probe)
    policy = PhysicalHands.install(env)
    tape = stance_recording(args.source) if args.pelvis == "free" else None
    trial = Trial(env, policy, args.hand, args.pelvis, tape, args.offset_mm)
    settings = resolved_trial(trial, args, meta, hashes, matched, probe)
    write_json(directory / "resolved.json", settings)
    result = run_probe(trial, probe) if probe else trial.run(matched)
    mujoco.mj_subtreeVel(env.model, env.data)
    result["end_momentum_Ns"] = (
        env.mass * env.data.subtree_linvel[env.model.body("pelvis").id, 2]
    )
    trial.readings.save(directory)
    np.savez_compressed(directory / "commands.npz", **trial.ticks)
    write_json(directory / "outcome.json", result)
    print(
        directory,
        result["end_s"],
        result["last_phase"],
        result["missing"] or "completed",
        flush=True,
    )
    return result


def prepare_scene(env, args, directory, probe):
    if probe == "wrong_surface":
        env.pose = replace(
            env.pose, hand_rungs=(6, 7) if args.hand == "left" else (7, 6)
        )
    if args.pelvis == "fixed":
        model, xml = fixed_model(env.xml, env.initial_qpos[:7])
        env.initialize_model(model)
        (directory / "fixture.xml").write_text(xml, encoding="utf-8")
    if probe == "wrong_surface":
        env.support_targets[args.hand + "_hand"] = "left_ladder_rung_07"


def stance_recording(source):
    with (source / "steps.jsonl").open() as stream:
        return np.array(
            [json.loads(line)["applied_position_targets_rad"] for line in stream]
        )


def resolved_trial(trial, args, meta, hashes, matched, probe):
    env, policy = trial.env, trial.policy
    return {
        "pelvis": args.pelvis,
        "side": args.hand,
        "offset_mm": args.offset_mm,
        "branch": "release" if matched is None else "control",
        "seed": meta["seed"],
        "config": asdict(env.config),
        "supports": SUPPORTS,
        "support_targets": env.support_targets,
        "states": STATES,
        "actuator_names": [env.model.actuator(i).name for i in range(env.model.nu)],
        "stance_actuators": policy.indices.tolist(),
        "geom_names": list(trial.readings.geom_ids),
        "mass_kg": env.mass,
        "gravity_z": env.model.opt.gravity[2],
        "initial_qpos": env.initial_qpos.tolist(),
        "source": str(args.source),
        "source_sha256": {
            n: hashes[n] for n in ("resolved.json", "scene.mjb", "steps.jsonl")
        },
        "source_model": meta["physical_identity"],
        "model": physical_identity(env.model),
        "controller": meta["controller"],
        "environment_spec": meta["environment_spec"],
        "grasp_pose": asdict(env.pose),
        "mujoco": mujoco.__version__,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runner_sha256": {
            p.name: file_hash(p) for p in Path(__file__).parent.glob("regrasp*.py")
        },
        "probe": probe,
        "grip_rate_per_s": 1.0,
        "ik_joint_travel_rad": 0.65,
        "ik_iteration_bound_rad": 0.12,
        "criteria": asdict(CRITERIA),
        "move_s": MOVE_S,
        "position_tolerance_m": POSITION_TOL_M,
        "rotation_tolerance_rad": ANGLE_TOL_RAD,
    }


def run_probe(trial, kind):
    trial.path = WristPath(trial.env, trial.policy, trial.side, 0.0)
    env, side = trial.env, trial.side
    start = env.target[trial.path.ids].copy()
    arm = start.copy()
    if kind == "air":
        roll = env.model.joint(side + "_shoulder_roll_joint")
        shift = 0.7 if side == "left" else -0.7
        arm[env.model.actuator_trnid[trial.path.ids, 0] == roll.id] += shift
        scratch = trial.path.scratch
        scratch.qpos[:] = env.data.qpos
        scratch.qpos[roll.qposadr[0]] += shift
        mujoco.mj_kinematics(env.model, scratch)
        trial.path.goal = rung_pose(env, scratch, side)[0]
    for _ in range(round(2.0 / trial.tick_s)):
        target = start + smooth(trial.time / MOVE_S) * (arm - start)
        trial.tick("prepare_probe", target, -1.0)
    trial.event("close_request")
    for _ in range(round(2.0 / trial.tick_s)):
        trial.tick("negative_close", arm, 1.0)
    return score_probe(trial, kind)


def score_probe(trial, kind):
    samples = trial.readings.samples
    mask = np.asarray(samples["phase"]) == "negative_close"
    states = np.asarray(samples["state"])[mask, trial.i]
    angles = np.asarray(samples["digits"])[mask, trial.i]
    records = np.concatenate(trial.readings.detail)
    owned = np.array(list(trial.readings.gap.hand))
    selected = np.isin(records[:, 1], owned) ^ np.isin(records[:, 2], owned)
    loaded = records[
        selected & mask[records[:, 0].astype(int)] & (records[:, 3] >= 0.05)
    ]
    others = np.where(np.isin(loaded[:, 1], owned), loaded[:, 2], loaded[:, 1]).astype(
        int
    )
    geoms = sorted({trial.env.model.geom(g).name for g in others})
    moved = float(np.max(angles[-1] - angles[0]))
    correct = (
        STATES.index("grasp") not in states
        and STATES.index(kind) in states
        and moved > 0.1
        and ((not geoms) if kind == "air" else "left_ladder_rung_06" in geoms)
    )
    return {
        "end_s": trial.time,
        "last_phase": kind,
        "missing": None if correct else "negative classification",
        "states": sorted({STATES[s] for s in states}),
        "loaded_surfaces": geoms,
        "measured_closing_rad": moved,
        "events": trial.events,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("--output", required=True, type=Path)
    trial = commands.add_parser("trial")
    trial.add_argument("--pelvis", choices=("fixed", "free"), required=True)
    trial.add_argument("--hand", choices=("left", "right"), required=True)
    trial.add_argument("--offset-mm", choices=(-2, 0, 2), type=int, default=0)
    trial.add_argument("--output", required=True, type=Path)
    negative = commands.add_parser("probe")
    negative.set_defaults(pelvis="fixed", offset_mm=0)
    negative.add_argument("--probe", choices=("air", "wrong_surface"), required=True)
    negative.add_argument("--hand", choices=("left", "right"), required=True)
    negative.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "reference":
        reproduce(args.source, args.output)
        return
    if args.command == "probe":
        run_branch(args, args.output)
        return
    result = run_branch(args, args.output / "release")
    if args.pelvis == "free" and "settled" in result["events"]:
        run_branch(args, args.output / "control", result)


if __name__ == "__main__":
    main()
