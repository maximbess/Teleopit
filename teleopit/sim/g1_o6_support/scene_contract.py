"""One resolved environment contract for hold, capacity and transfer evidence."""

import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.grasp_pose import GraspPose
from teleopit.sim.g1_o6.material import Material
from teleopit.sim.g1_o6.metadata import resolved_controller
from teleopit.sim.ladder_config import LadderConfig, compiled_ladder

from .ladder_identity import assert_preserved, physical_identity
from .g6_controllers import controller_from_spec, environment_factory


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def environment_metadata(env):
    result = {
        "environment_spec": {
            "ladder": asdict(env.ladder) if env.ladder else None,
            "material": asdict(env.material) if env.material else None,
            "adapt_reset": env.adapt_reset,
        }
    }
    if env.ladder is not None:
        result.update(
            ladder=compiled_ladder(env.model, env.ladder),
            material=env.audit.get("material"),
            physical_identity=physical_identity(env.model),
            required_contact_diagnostics=True,
        )
    if hasattr(env, "arm_feedback"):
        result["environment_spec"]["arm_feedback"] = asdict(env.arm_feedback)
    if hasattr(env, "grasp_acquisition"):
        result["environment_spec"]["grasp_acquisition"] = asdict(env.grasp_acquisition)
    return result


def restore_environment(metadata, physics_dt, *, saved_scene=None):
    spec = metadata.get("environment_spec")
    if spec is None:
        raise ValueError("resolved environment specification is required")
    factory, options = environment_factory(controller_from_spec(spec))
    env = factory(
        Config(**{**metadata["config"], "physics_dt": physics_dt}),
        pose=GraspPose(**metadata["grasp_pose"]),
        ladder=LadderConfig(**spec["ladder"]) if spec["ladder"] else None,
        material=Material(**spec["material"]) if spec["material"] else None,
        adapt_reset=spec["adapt_reset"],
        **options,
    )
    if saved_scene is not None and Path(saved_scene).is_file() and "physical_identity" in metadata:
        archived = mujoco.MjModel.from_binary_path(str(saved_scene))
        reference = metadata["physical_identity"]
        assert_preserved(physical_identity(archived), reference)
        # Absolute mesh filename lengths affect this byte-offset table. Compare every
        # physical field of the rebuild, then bind the actual verified source binary.
        assert_preserved(physical_identity(env.model, mesh_pathadr=archived.mesh_pathadr), reference)
        archived.opt.timestep = physics_dt
        env.initialize_model(archived)
    verify_environment(env, metadata)
    return env


def verify_environment(env, metadata):
    actual = environment_metadata(env)
    for key in ("environment_spec", "ladder", "material"):
        if canonical(actual.get(key)) != canonical(metadata.get(key)):
            raise ValueError("compiled environment differs: " + key)
    if "physical_identity" in metadata:
        assert_preserved(physical_identity(env.model), metadata["physical_identity"])
    if canonical(resolved_controller(env)) != metadata["controller"]:
        raise ValueError("unsupported or changed source controller")
    initial = metadata["initial"]
    np.testing.assert_array_equal(
        env.data.qpos, [*initial["base_pose"], *initial["joint_positions"]]
    )


def capacity_source_metadata(metadata):
    keys = ("ladder", "material", "physical_identity", "environment_spec", "initial")
    return {"source_" + key: metadata[key] for key in keys if key in metadata}
