"""Regression checks for the ladder-only OmniRetarget collision overlay."""
import json
import re
import xml.etree.ElementTree as ET

import mujoco
import pytest
import torch  # Import before mjlab, matching the training test environment.
import numpy as np
from mjlab.entity import Entity
from scipy.spatial.transform import Rotation

from teleopit.runtime.g1_collision_assets import ASSET_DIR, BODY_MAP, CANONICAL_JOINT_ORIGIN_DELTAS
from train_mimic.tasks.tracking.config import env as config
from train_mimic.tasks.tracking.config.g1_collision import apply_g1_collision_overlay


def test_donor_moving_frames_match_canonical_g1():
    """Copying mesh coordinates is safe only when all moving frames agree."""
    donor = ET.parse(ASSET_DIR / "source/g1_29dof.urdf").getroot()
    canonical = config._get_g1_training_spec().compile()
    joints = {joint.find("child").get("link"): joint for joint in donor.findall("joint")}
    for body_id in range(1, canonical.nbody):
        name = canonical.body(body_id).name
        if name not in joints:
            continue
        joint = joints[name]
        assert joint.find("parent").get("link") == canonical.body(canonical.body_parentid[body_id]).name
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        expected_delta = CANONICAL_JOINT_ORIGIN_DELTAS.get(name, (0.0, 0.0, 0.0))
        np.testing.assert_allclose(canonical.body_pos[body_id], xyz + expected_delta, atol=1e-6, rtol=0, err_msg=name)
        canonical_rotation = Rotation.from_quat(canonical.body_quat[body_id][[1, 2, 3, 0]])
        np.testing.assert_allclose(canonical_rotation.as_matrix(), Rotation.from_euler("xyz", rpy).as_matrix(), atol=1e-6, rtol=0, err_msg=name)
        if joint.get("type") != "fixed":
            joint_id = canonical.joint(joint.get("name")).id
            np.testing.assert_allclose(canonical.jnt_axis[joint_id], np.fromstring(joint.find("axis").get("xyz"), sep=" "), atol=1e-8)


def initial_state(model, cfg):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:3] = cfg.init_state.pos
    data.qpos[3:7] = cfg.init_state.rot
    for j in range(model.njnt):
        name = model.joint(j).name
        for pattern, value in cfg.init_state.joint_pos.items():
            if re.fullmatch(pattern, name):
                data.qpos[model.jnt_qposadr[j]] = value
    mujoco.mj_forward(model, data)
    return data


def test_overlay_preserves_canonical_dynamics_and_sites(monkeypatch):
    cfg = config.make_g1_ladder_training_robot_cfg()
    refined = Entity(cfg).spec.compile()
    with monkeypatch.context() as context:
        context.setattr(config, "apply_g1_collision_overlay", lambda spec: None)
        original_cfg = config.make_g1_ladder_training_robot_cfg()
        original_cfg.collisions = tuple(c for c in original_cfg.collisions if not any("foot_omni" in pattern for pattern in c.geom_names_expr))
        original = Entity(original_cfg).spec.compile()
    assert (refined.nq, refined.nv, refined.nu) == (original.nq, original.nv, original.nu) == (36, 35, 29)
    for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
                 "jnt_pos", "jnt_axis", "jnt_range", "dof_damping", "qpos0", "actuator_gear",
                 "actuator_forcerange", "site_pos", "site_quat"):
        np.testing.assert_allclose(getattr(refined, name), getattr(original, name), atol=1e-10, rtol=0, err_msg=name)


def test_overlay_checks_missing_and_corrupt_assets(tmp_path):
    spec = config._get_g1_training_spec()
    with pytest.raises(FileNotFoundError, match="g1_collision"):
        apply_g1_collision_overlay(spec, tmp_path)
    manifest = json.loads((ASSET_DIR / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    count = len(list(spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)))
    with pytest.raises(ValueError, match="Missing or corrupt"):
        apply_g1_collision_overlay(spec, tmp_path)
    assert len(list(spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM))) == count


def test_refined_initial_contacts_and_mesh_budget():
    cfg = config.make_g1_ladder_training_robot_cfg()
    model = Entity(cfg).spec.compile()
    data = initial_state(model, cfg)
    names = [model.geom(i).name for i in range(model.ngeom)]
    assert not any("blocker" in name for name in names)
    refined_ids = [i for i, name in enumerate(names) if "_omni_" in name]
    assert refined_ids
    assert len(refined_ids) <= len(BODY_MAP) * 12
    for i in refined_ids:
        assert model.geom_contype[i] == model.geom_conaffinity[i] == 1
        assert model.mesh_vertnum[model.geom_dataid[i]] <= 64
    contacts = []
    foot_sides = set()
    for c in data.contact:
        pair = [names[int(i)] for i in c.geom]
        contacts.append((pair, float(c.dist)))
        if "left_ladder_rung_02" in pair:
            for side in ("left", "right"):
                if any(name.startswith(side + "_foot_omni_") for name in pair):
                    foot_sides.add(side)
                    assert c.dim == 4
                    assert c.friction[0] == pytest.approx(1.8)
    print("Initial contacts:", contacts)
    assert foot_sides == {"left", "right"}
    assert all(distance >= -0.003 for _, distance in contacts), contacts
    assert all(any("_foot_omni_" in name for name in pair) for pair, _ in contacts), contacts


def test_unwanted_contact_sensors_compile_with_both_faces():
    cfg = config.make_g1_ladder_rl_env_cfg()
    robot = Entity(cfg.scene.entities["robot"])
    scene = mujoco.MjSpec()
    scene.attach(robot.spec, prefix="robot/", frame=scene.worldbody.add_frame())
    sensors = [s.build() for s in cfg.scene.sensors
               if s.name.startswith("ladder_unwanted_contact_")]
    assert len(sensors) == 2
    for sensor in sensors:
        sensor.edit_spec(scene, {"robot": robot})
    assert sensors[0].primary_names == sensors[1].primary_names
    assert "pelvis" in sensors[0].primary_names
    assert "torso_link" in sensors[0].primary_names
    assert "left_knee_link" in sensors[0].primary_names
    assert "left_elbow_link" in sensors[0].primary_names
    assert not set(sensors[0].primary_names) & set(sensors[0].cfg.primary.exclude)
    model = scene.compile()
    reference_ids = {int(model.sensor_refid[i]) for i in range(model.nsensor)
                     if model.sensor(i).name.startswith("ladder_unwanted_contact_")}
    assert reference_ids == {model.body("robot/left_ladder_body").id,
                             model.body("robot/right_ladder_body").id}


def test_ladder_convex_contacts_have_zero_margin_for_warp_multiccd():
    cfg = config.make_g1_ladder_training_robot_cfg()
    # Check both assembly layers: the stock collision editor can overwrite
    # values emitted by the scene builder before Warp receives the model.
    for model in (cfg.spec_fn().compile(), Entity(cfg).spec.compile()):
        ids = [i for i in range(model.ngeom)
               if model.geom_contype[i] or model.geom_conaffinity[i]]
        convex_ids = [i for i in ids if model.geom_type[i] in (
            mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH)]
        assert convex_ids
        np.testing.assert_array_equal(model.geom_margin[convex_ids], 0.0)
        np.testing.assert_array_equal(model.pair_margin, 0.0)


def test_collision_surfaces_align_with_canonical_visuals():
    cfg = config.make_g1_ladder_training_robot_cfg()
    model = Entity(cfg).spec.compile()
    data = initial_state(model, cfg)
    manifest = json.loads((ASSET_DIR / "manifest.json").read_text())

    def vertices(geom_id):
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        points = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
        return points @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]

    for source in BODY_MAP:
        visual_mesh = model.mesh(source).id
        visual_id = next(i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_dataid[i] == visual_mesh)
        expected = vertices(visual_id)
        ids = [i for i, part in enumerate(manifest["parts"]) if part["source_body"] == source]
        actual = np.concatenate([vertices(next(j for j in range(model.ngeom) if model.geom_dataid[j] == model.mesh(f"omni_g1_{i:03d}").id and model.geom_type[j] == mujoco.mjtGeom.mjGEOM_MESH)) for i in ids])
        np.testing.assert_allclose(actual.min(axis=0), expected.min(axis=0), atol=0.006, rtol=0, err_msg=source)
        np.testing.assert_allclose(actual.max(axis=0), expected.max(axis=0), atol=0.006, rtol=0, err_msg=source)
