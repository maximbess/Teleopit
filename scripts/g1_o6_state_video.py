"""Offline rendering of recorded qpos; never advances or qualifies physics."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from teleopit.sim.g1_o6.viewer import camera


def samples(path, tail):
    summary = json.loads((path / "summary.json").read_text())
    end = summary["strict"]["events"][-1]["time"]
    selected = []
    with (path / "steps.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["time"] >= end - tail:
                selected.append(row)
    return selected[::5] + [selected[-1]]


def replay_frame(model, data, renderer, options, row, cams, label):
    data.qpos[:] = [*row["base_pose"], *row["joint_positions"]]
    data.ctrl[:] = row["applied_position_targets_rad"]
    mujoco.mj_forward(model, data)  # Offline display data, no mj_step.
    images = []
    for cam in cams:
        renderer.update_scene(data, camera=cam, scene_option=options)
        images.append(renderer.render().copy())
    picture = Image.fromarray(np.concatenate(images, axis=1))
    draw = ImageDraw.Draw(picture)
    draw.rectangle((0, 0, picture.width, 42), fill="black")
    draw.text((8, 4), f"Recorded state replay | {label}", fill="white")
    draw.text(
        (8, 23),
        f"t={row['time']:.2f} s | RESET drift={1000 * row['base_drift_m']:.2f} mm | assistance={row['assistance_gain']:g}",
        fill="orange",
    )
    return np.asarray(picture)


def video(run, output, tail):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    metadata = json.loads((run / "resolved.json").read_text())
    status = json.loads((run / "summary.json").read_text())["strict_status"]
    data = mujoco.MjData(model)
    frames = samples(run, tail)
    cameras = [mujoco.MjvCamera(), mujoco.MjvCamera()]
    camera(cameras[0])
    camera(cameras[1])
    data.qpos[:] = [*frames[0]["base_pose"], *frames[0]["joint_positions"]]
    mujoco.mj_forward(model, data)
    cameras[1].lookat[:] = np.mean(
        [
            data.site(metadata["model"]["hands"][s]["mount"]).xpos
            for s in ("left", "right")
        ],
        axis=0,
    )
    cameras[1].distance = 1.0
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    with (
        mujoco.Renderer(model, height=480, width=640) as renderer,
        imageio.get_writer(output / "replay.mp4", fps=10) as writer,
    ):
        for i, row in enumerate(frames):
            pixels = replay_frame(
                model,
                data,
                renderer,
                options,
                row,
                cameras,
                run.name + " | strict result " + status,
            )
            writer.append_data(pixels)
            if i in (0, len(frames) - 1):
                imageio.imwrite(
                    output / ("first.png" if i == 0 else "last.png"), pixels
                )
    (output / "replay.json").write_text(
        json.dumps(
            {
                "source_run": str(run),
                "method": "recorded qpos, offline MjData, forward for display only; no physics steps",
                "first_time_s": frames[0]["time"],
                "last_time_s": frames[-1]["time"],
                "fps": 10,
                "frames": len(frames),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-seconds", type=float, default=10)
    args = parser.parse_args()
    video(args.run, args.output, args.tail_seconds)
