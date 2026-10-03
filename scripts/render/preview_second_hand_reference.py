"""Retarget, validate and preview right-hand motion from a terminal pose bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from scripts.render import preview_ladder_reference as preview
from teleopit.runtime.cartesian_reference import load_reference, solve_reference
from teleopit.runtime.second_hand_reference import build_second_hand_spec, assess_second_hand
from train_mimic.scripts.preview_terminal_poses import restore_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=ROOT / "outputs/first_hand_pose_bank")
    parser.add_argument("--design", type=Path, default=ROOT / "assets/motions/ladder_second_hand.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/second_hand_reference")
    parser.add_argument("--index", type=int, help="Default: bank medoid")
    parser.add_argument("--all-representatives", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--paused", action="store_true")
    args = parser.parse_args()
    meta = json.loads((args.bank / "manifest.json").read_text(encoding="utf-8"))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    nominal = meta["nominal_id"] if args.index is None else args.index
    ids = list(dict.fromkeys([nominal, *(meta["representatives"] if args.all_representatives else [])]))
    model = mujoco.MjModel.from_binary_path(str(args.bank / "scene.mjb"))
    data = mujoco.MjData(model)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    nominal_result = None
    for index in ids:
        record = next((r for r in meta["records"] if r["id"] == index), None)
        if record is None:
            parser.error(f"Pose {index} absent from bank")
        source = args.bank / record["file"]
        with np.load(source, allow_pickle=False) as f:
            state = {k: f[k] for k in f.files}
        restore_pose(model, data, state)
        initial = data.qpos.copy()
        spec = build_second_hand_spec(model, data, state, meta, design)
        spec["provenance"] = {"pose_id": index, "state_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "checkpoint_reference_signature": meta["checkpoint_reference_signature"],
            "runtime_reference_signature": meta["runtime_reference_signature"],
            "reference_signature_matches": meta["reference_signature_matches"],
            "source": "Measured local policy execution; consult reference_signature_matches for checkpoint compatibility.",
            "initial_measured_qvel_max_abs": float(np.abs(state["qvel"][-1]).max()),
            "initial_velocity_note": "Reference starts at rest in the measured pose. Blend measured velocity during settle in a future physical controller."}
        out = args.output / f"pose_{index:04d}"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "reference.json"
        path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        spec, tracks = load_reference(path, model, data)
        print(f"Solving pose {index} ({spec['duration_s']} s)...", flush=True)
        # No enabled weld may be mistaken for dynamic execution: this is IK only.
        data.eq_active[:] = False
        clip, report = solve_reference(model, data, spec, tracks,
                                       posture_weight=design["posture_weight"], freeze_static_targets=True)
        extra, velocity = assess_second_hand(model, data, spec, clip, initial)
        report.update(extra)
        limits = design["quality_limits"]
        report["quality_limits"] = limits
        report["motion_quality_ok"] = bool(
            extra["minimum_joint_limit_margin_rad"] >= limits["joint_margin_rad"]
            and extra["max_joint_speed_rad_s"] <= limits["joint_speed_rad_s"]
            and extra["max_joint_acceleration_rad_s2"] <= limits["joint_acceleration_rad_s2"])
        report["kinematic_checks_ok"] = bool(report["tracking_ok"] and not report["collision_frames"]
            and report["release_orientation_ok"] and report["joint_limits_ok"] and report["finite"]
            and report["initial_qpos_max_error"] < 1e-6 and report["motion_quality_ok"])
        report["validation_scope"] = "Each pose is independently retargeted. Does not test a fixed clip across starts, force feasibility, grip control or policy robustness."
        report["final_hold_max_joint_speed_rad_s"] = float(np.abs(velocity[-10:,6:]).max())
        np.savez_compressed(out / "reference.npz", **clip, qvel=velocity,
                            fps=spec["fps"], joint_qposadr=model.jnt_qposadr,
                            joint_dofadr=model.jnt_dofadr, initial_measured_qvel=state["qvel"][-1],
                            joint_names=np.array(meta["physics_joint_names"]),
                            track_names=np.array(meta["track_names"]), reference_json=json.dumps(spec))
        (out / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary[str(index)] = {k:v for k,v in report.items() if k != "collision_frames"}
        summary[str(index)]["collision_frame_count"] = len(report["collision_frames"])
        print(json.dumps(summary[str(index)], indent=2), flush=True)
        if index == nominal:
            nominal_result = spec, tracks, clip, report, out, state
    (args.output / "validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    spec, tracks, clip, report, out, state = nominal_result
    restore_pose(model, data, state)
    def camera_settings(camera):
        camera.lookat[:] = state["track_pos"][-1,4] + [0.,0.,.25]
        camera.distance = 3.2
        camera.azimuth = 30.
        camera.elevation = -8.
    preview.camera_settings = camera_settings
    if args.render or args.video:
        preview.render(model, data, spec, tracks, clip, out, video=args.video)
    if not args.headless:
        preview.play(model, data, spec, tracks, clip, report, paused=args.paused)
    return 0 if all(r["kinematic_checks_ok"] for r in summary.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
