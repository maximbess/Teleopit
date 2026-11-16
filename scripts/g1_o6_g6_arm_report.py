"""Audit bounded arm feedback from actual commands and saved-model kinematics."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.g1_o6_g6_report import read, report


def command_pose(model, data, mount, indices, command):
    addresses = model.jnt_qposadr[model.actuator_trnid[indices, 0]]
    data.qpos[addresses] = command[indices]
    mujoco.mj_kinematics(model, data)
    site = data.site(mount)
    return site.xpos.copy(), site.xmat.reshape(3, 3).copy()


def command_sample(model, mounts, indices, baseline, previous, row, dt):
    data = mujoco.MjData(model)
    command = np.asarray(row["applied_position_targets_rad"])
    translation, rotation = [], []
    for side, selected in indices.items():
        data.qpos[:] = [*row["base_pose"], *row["joint_positions"]]
        origin = command_pose(model, data, mounts[side], selected, baseline)
        current = command_pose(model, data, mounts[side], selected, command)
        translation.append(float(np.linalg.norm(current[0] - origin[0])))
        rotation.append(float(Rotation.from_matrix(current[1] @ origin[1].T).magnitude()))
    arms = np.concatenate(list(indices.values()))
    return {"time": row["time"], "translation_m": max(translation),
            "rotation_rad": max(rotation),
            "offset_rad": float(np.max(abs(command[arms] - baseline[arms]))),
            "rate_rad_s": float(np.max(abs(command[arms] - previous[arms])) / dt)}


def assert_bounds(sample, rate):
    # Independent experimental requirements, not self-reported controller state.
    for name, limit in (("translation_m", .005), ("rotation_rad", .03),
                        ("offset_rad", .03), ("rate_rad_s", rate)):
        assert np.isfinite(sample[name]) and sample[name] <= limit + 1e-8, (name, sample)


def arm_mapping(model, metadata):
    indices = {}
    for side, names in metadata["controller"]["arm_joint_order"].items():
        indices[side] = np.array([i for i, joint in enumerate(model.actuator_trnid[:, 0])
                                  if model.joint(joint).name in names])
        assert len(indices[side]) == len(names) == 7
    return indices


def verify_run(path, output):
    metadata = read(path / "resolved.json")
    controller = metadata["controller"]
    if controller.get("mode") != "g6_acquisition_arm_feedback":
        return {"status": "NOT_APPLICABLE", "reason": "unchanged reference controller"}
    model = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
    mounts = {side: value["mount"] for side, value in metadata["model"]["hands"].items()}
    indices = arm_mapping(model, metadata)
    rate = controller["offset_rate_rad_s"]
    assert rate in (.005, .02)
    samples = []
    with (path / "steps.jsonl").open() as stream:
        first = json.loads(next(stream))
        baseline = np.asarray(first["applied_position_targets_rad"])
        previous, previous_time = baseline, first["time"]
        for line in stream:
            row = json.loads(line)
            sample = command_sample(model, mounts, indices, baseline, previous, row, row["time"] - previous_time)
            assert_bounds(sample, rate)
            samples.append(sample)
            previous, previous_time = np.asarray(row["applied_position_targets_rad"]), row["time"]
    assert samples, "empty command trace"
    output.write_text(json.dumps(samples, indent=2) + "\n")
    return {"status": "PASS", "method": "saved MJB, logged applied commands, independent scratch FK; no controller-state claims",
            "samples": len(samples), "declared_rate_rad_s": rate,
            "maxima": {key: max(row[key] for row in samples) for key in samples[0] if key != "time"}}


def verify(root):
    report(root)
    state = read(root / "campaign_state.json")
    output = root / "report"
    checks = {record["id"]: verify_run(root / record["output"], output / (record["id"] + "_commands.json"))
              for record in state["runs"]}
    (output / "command_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/g1_o6/g6_arm_feedback"))
    verify(parser.parse_args().root)
