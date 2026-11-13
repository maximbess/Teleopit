"""Front/side compiled-scene inspection, with actual dimensions and reset pose."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.grasp_pose import GraspPose
from teleopit.sim.ladder_config import compiled_ladder, geometry_matrix


def views(root):
    state = json.loads((root / "campaign_state.json").read_text())
    output = root / "geometry_views"
    output.mkdir(exist_ok=True)
    for identity, ladder in geometry_matrix().items():
        reset = next(
            d["value"] for d in state["decisions"] if d["name"] == "reset_" + identity
        )
        if reset["pose"] is None:
            continue
        env = Environment(ladder=ladder, pose=GraspPose(**reset["pose"]))
        report = compiled_ladder(env.model, ladder)
        (output / (identity + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        pictures = []
        with mujoco.Renderer(env.model, height=720, width=600) as renderer:
            for azimuth in (180, 90):
                camera = mujoco.MjvCamera()
                camera.lookat[:] = (0, 0, max(1.3, ladder.apex_height_m / 2))
                camera.distance = max(4.5, ladder.apex_height_m * 1.7)
                camera.azimuth, camera.elevation = azimuth, 0
                renderer.update_scene(env.data, camera=camera, scene_option=options)
                pictures.append(renderer.render().copy())
        picture = Image.fromarray(np.concatenate(pictures, axis=1))
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, 1200, 80), fill="black")
        draw.text(
            (12, 8),
            f"{identity} | FRONT (left) / SIDE (right) | compiled reset; no physics stepped",
            fill="white",
        )
        draw.text(
            (12, 27),
            f"Tread L x D x T: {ladder.length_m * 1000:g} x {ladder.depth_m * 1000:g} x {ladder.thickness_m * 1000:g} mm | incline {ladder.inclination_deg:.3f} deg",
            fill="white",
        )
        draw.text(
            (12, 46),
            f"Pitch vertical {report['front_pitch']['vertical_m'] * 1000:.3f} mm; along rail {report['front_pitch']['along_rail_m'] * 1000:.3f} mm | hands {reset['selected']['hand']}; feet {reset['selected']['foot']}",
            fill="white",
        )
        if identity == "G6":
            draw.text(
                (12, 63),
                "Commercial stepladder approximation; only depth and vertical pitch are product-family specifications",
                fill="orange",
            )
        picture.save(output / (identity + ".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/g1_o6/ladder_material")
    )
    views(parser.parse_args().output)
