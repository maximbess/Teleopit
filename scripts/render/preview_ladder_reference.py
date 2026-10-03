#!/usr/bin/env python3
"""Preview authored multi-body ladder targets and their IK solution in MuJoCo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from teleopit.runtime.cartesian_reference import load_reference, solve_reference
from teleopit.runtime.ladder_scene import build_ladder_preview_model


def camera_settings(camera):
    camera.lookat[:] = (-.62, 0, 1.24)
    camera.distance = 2.85
    camera.azimuth = -35
    camera.elevation = -12


def show_options(option):
    option.geomgroup[:] = 1
    option.geomgroup[3] = 0  # Collision meshes remain active, just hidden.
    option.sitegroup[:] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False


def add_sphere(scene, pos, color, radius=.018, label=""):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius),
                       np.array(pos), np.eye(3).ravel(), np.array(color, dtype=np.float32))
    geom.label = label
    scene.ngeom += 1


def add_line(scene, a, b, color, radius=.003):
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(b-a) < 1e-8:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                       np.zeros(3), np.eye(3).ravel(), np.array(color, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
    scene.ngeom += 1


def overlay(scene, clip, tracks, frame, paths=True):
    for i, track in enumerate(tracks):
        target = clip["target_position"][frame, i]
        actual = clip["actual_position"][frame, i]
        if paths:
            path = clip["target_position"][:, i]
            indices = np.unique(np.r_[np.arange(0, len(path), max(1, len(path)//100)), len(path)-1])
            for a, b in zip(indices[:-1], indices[1:]):
                add_line(scene, path[a], path[b], track.color)
            for pos in track.positions:
                add_sphere(scene, pos, track.color, .009)
        add_sphere(scene, target, track.color, .018)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, clip["target_quaternion_wxyz"][frame, i])
        for axis, color in zip(rotation.reshape(3, 3).T,
                               ([1, .25, .25, 1], [.25, 1, .25, 1], [.25, .45, 1, 1])):
            add_line(scene, target, target + .035*axis, color, .0015)
        error = clip["position_error_m"][frame, i]
        if error > .005:
            add_line(scene, actual, target, [1, .1, .1, 1], .005)
            add_sphere(scene, actual, [1, .1, .1, 1], .014)


def frame_text(spec, clip, tracks, frame, report, paused, speed):
    t = clip["time_s"][frame]
    phase = next((p["name"] for p in reversed(spec.get("phases", [])) if t >= p["time_s"]), "Reference")
    lines = ["KINEMATIC REFERENCE / no dynamics", f"{t:.2f} / {spec['duration_s']:.2f} s | {phase}",
             f"{'PAUSED' if paused else 'PLAY'} | {speed:.2f}x", ""]
    lines += [f"{track.name}: {clip['position_error_m'][frame,i]*1000:.2f} mm / "
              f"{np.rad2deg(clip['orientation_error_rad'][frame,i]):.2f} deg"
              for i, track in enumerate(tracks)]
    lines += ["", "P: play/pause | J/L: step | R: restart", "[ / ]: speed | T: paths | C: camera",
              "Orange: left hand | Blue: right hand", "Green/yellow: feet | Purple/pink: pelvis/torso",
              "Red line: target error > 5 mm"]
    if not report["tracking_ok"]:
        lines.append("WARNING: reference contains unresolved IK targets")
    if report["collision_frames"]:
        lines.append(f"WARNING: penetration in {len(report['collision_frames'])} frames; see report")
    if report.get("kinematic_checks_ok") is False:
        lines.append("WARNING: kinematic acceptance checks failed; see validation.json")
    return "\n".join(lines)


def apply_frame(model, data, clip, frame):
    data.qpos[:] = clip["qpos"][frame]
    data.qvel[:] = 0
    data.eq_active[:] = False
    data.time = clip["time_s"][frame]
    mujoco.mj_forward(model, data)


def play(model, data, spec, tracks, clip, report, *, paused=False, speed=.5):
    import mujoco.viewer
    import glfw
    from queue import SimpleQueue
    commands = SimpleQueue()
    with mujoco.viewer.launch_passive(model, data, key_callback=commands.put,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        camera_settings(viewer.cam)
        show_options(viewer.opt)
        cursor, paths, last = 0., True, time.monotonic()
        print("MuJoCo reference viewer ready. P: play/pause, J/L: step, R: restart.", flush=True)
        # Hold the last frame; R restarts without hiding an end-to-start jump.
        while viewer.is_running():
            now = time.monotonic()
            elapsed, last = now-last, now
            while not commands.empty():
                key = commands.get()
                if key == glfw.KEY_P:
                    paused = not paused
                elif key in (glfw.KEY_J, glfw.KEY_L):
                    paused = True
                    cursor += -1 if key == glfw.KEY_J else 1
                elif key == glfw.KEY_R:
                    cursor = 0
                elif key == glfw.KEY_LEFT_BRACKET:
                    speed = max(.125, speed/2)
                elif key == glfw.KEY_RIGHT_BRACKET:
                    speed = min(4, speed*2)
                elif key == glfw.KEY_T:
                    paths = not paths
                elif key == glfw.KEY_C:
                    camera_settings(viewer.cam)
            if not paused:
                cursor += elapsed*spec["fps"]*speed
            cursor = float(np.clip(cursor, 0, len(clip["time_s"])-1))
            frame = int(cursor)
            if frame == len(clip["time_s"])-1:
                paused = True
            with viewer.lock():
                apply_frame(model, data, clip, frame)
                viewer.user_scn.ngeom = 0
                overlay(viewer.user_scn, clip, tracks, frame, paths)
            if hasattr(viewer, "set_texts"):
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                                  mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  frame_text(spec, clip, tracks, frame, report, paused, speed), ""))
            viewer.sync()
            time.sleep(.01)


def render(model, data, spec, tracks, clip, out, *, video=False):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    camera = mujoco.MjvCamera()
    camera_settings(camera)
    option = mujoco.MjvOption()
    show_options(option)
    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        writer = imageio.get_writer(out / "preview.mp4", fps=spec["fps"]/2, codec="libx264") if video else None
        try:
            frames = range(0, len(clip["time_s"]), 2) if video else [0, 105, len(clip["time_s"])-1]
            for frame in frames:
                frame = min(frame, len(clip["time_s"])-1)
                apply_frame(model, data, clip, frame)
                renderer.update_scene(data, camera=camera, scene_option=option)
                overlay(renderer.scene, clip, tracks, frame)
                pixels = renderer.render()
                labeled = Image.fromarray(pixels)
                draw = ImageDraw.Draw(labeled)
                phase = next((p["name"] for p in reversed(spec.get("phases", []))
                              if clip["time_s"][frame] >= p["time_s"]), "Reference")
                draw.rectangle((12, 12, 455, 77), fill=(25, 30, 38))
                draw.text((23, 21), "KINEMATIC REFERENCE | Not dynamically validated", fill="white")
                draw.text((23, 43), f"{clip['time_s'][frame]:.2f} s | {phase}", fill=(255, 185, 85))
                pixels = np.asarray(labeled)
                if writer:
                    writer.append_data(pixels)
                if frame in (0, 104, 105, len(clip["time_s"])-1):
                    imageio.imwrite(out / f"frame_{frame:03d}.png", pixels)
        finally:
            if writer:
                writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "assets/motions/ladder_first_hand.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ladder_reference_preview")
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument("--headless", action="store_true", help="Solve/export without opening a viewer")
    parser.add_argument("--render", action="store_true", help="Save representative MuJoCo frames")
    parser.add_argument("--video", action="store_true", help="Save an offscreen MP4")
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--speed", type=float, default=.5)
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive and finite")
    model, data = build_ladder_preview_model(args.robot_xml)
    spec, tracks = load_reference(args.reference, model, data)
    print(f"Solving {len(tracks)} tracks: {spec['name']}", flush=True)
    clip, report = solve_reference(model, data, spec, tracks)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "reference_preview.npz", **clip,
                        fps=spec["fps"], track_names=np.array([t.name for t in tracks]),
                        reference_json=json.dumps(spec))
    (args.output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "collision_frames"}, indent=2), flush=True)
    print(f"Collision frames: {len(report['collision_frames'])}; outputs: {args.output}", flush=True)
    if args.render or args.video:
        render(model, data, spec, tracks, clip, args.output, video=args.video)
    if not args.headless:
        play(model, data, spec, tracks, clip, report, paused=args.paused, speed=args.speed)


if __name__ == "__main__":
    main()
