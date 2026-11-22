"""Loop a completed rung-7 release and regrasp in Viser using the recorded joint positions."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from teleopit.sim.g1_o6_support.ladder_campaign import file_hash

ROOT = Path("outputs/g1_o6/rung7_free_regrasp")


def cycle_frames(clock, outcome):
    events = outcome["events"]
    if outcome["missing"] or "recovered2" not in events:
        raise ValueError("A continuous, completed moving-hand cycle is required.")
    return (clock >= events["open_request1"] - 1e-8) & (
        clock <= events["recovered2"] + 1e-8
    )


def read_clip(run):
    cfg = json.loads((run / "resolved.json").read_bytes())
    outcome = json.loads((run / "outcome.json").read_bytes())
    if cfg["pelvis"] == "fixed":
        model = mujoco.MjModel.from_xml_path(str(run / "fixture.xml"))
    else:
        scene = Path(cfg["source"]) / "scene.mjb"
        if file_hash(scene) != cfg["source_sha256"]["scene.mjb"]:
            raise ValueError("The scene differs from the recorded model.")
        model = mujoco.MjModel.from_binary_path(str(scene))
    with np.load(run / "commands.npz") as data:
        frames = cycle_frames(data["time"], outcome)
        clock, phase, qpos = (data[key][frames] for key in ("time", "phase", "qpos"))
    if (
        len(clock) < 2
        or qpos.shape != (len(clock), model.nq)
        or not np.isfinite(qpos).all()
    ):
        raise ValueError("Incomplete recording or joint layout mismatch.")
    return model, cfg["side"], clock, phase, qpos


def show(model, side, clock, phase, qpos, port):
    import viser
    from mjviser import ViserMujocoScene

    server = viser.ViserServer(
        host="127.0.0.1", port=port, label=f"Rung 7 · {side} hand"
    )
    data = mujoco.MjData(model)
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = False
    scene.site_groups_visible = [False] * 6
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    focus = data.site(side + "_o6_wrist_mount").xpos.copy()
    lateral = 1 if side == "left" else -1

    @server.on_client_connect
    def camera(client):
        client.camera.up_direction = (0, 0, 1)
        client.camera.position = focus + [-0.55, lateral * 0.8, 0.18]
        client.camera.look_at = focus

    play = server.gui.add_checkbox("Play", initial_value=True)
    speed = server.gui.add_dropdown(
        "Speed", ("0.25", "0.5", "1", "2"), initial_value="1"
    )
    frame = server.gui.add_slider(
        "Frame", min=0, max=len(clock) - 1, step=1, initial_value=0
    )
    label = server.gui.add_markdown("")
    try:
        while True:
            i = int(frame.value)
            data.qpos[:] = qpos[i]
            mujoco.mj_kinematics(model, data)
            scene.update_from_mjdata(data)
            label.content = (
                f"**{phase[i]}** · {clock[i] - clock[0]:.2f} s from the open request"
            )
            time.sleep((clock[1] - clock[0]) / float(speed.value))
            if play.value and frame.value == i:
                frame.value = (i + 1) % len(clock)
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("left", "right"), default="left")
    parser.add_argument(
        "--run", type=Path, help="Override the default nominal release directory"
    )
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()
    run = args.run or ROOT / f"free_{args.hand}_1" / "release"
    show(*read_clip(run), args.port)


if __name__ == "__main__":
    main()
