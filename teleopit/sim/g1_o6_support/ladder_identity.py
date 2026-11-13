"""Separate unchanged robot/solver/controller identity from ladder and pairs."""

import hashlib
import subprocess
from pathlib import Path

import numpy as np

from teleopit.sim.g1_o6.diagnostics import region

REFERENCE = "cc0cd5393c670e7801bb7d242c700e6af4d47303"


def array_digest(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def physical_identity(model):
    bodies = [b for b in range(model.nbody) if region(model, b) != "environment"]
    geoms = [g for g in range(model.ngeom) if "_ladder_" not in model.geom(g).name]
    selected = []
    for name in dir(model):
        value = getattr(model, name)
        if not isinstance(value, np.ndarray):
            continue
        physical = invariant_field(model, name, value, bodies, geoms)
        if physical is not None:
            selected.append((name, physical))
    solver = solver_fields(model.opt)
    return {
        "robot_and_ground_sha256": array_digest(selected),
        "solver_sha256": array_digest(solver),
        "controller_sha256": controller_hash(),
        "physics_dt": float(model.opt.timestep),
    }


def invariant_field(model, name, value, bodies, geoms):
    if name.startswith("body_") and len(value) == model.nbody:
        return value[bodies]
    if name.startswith("geom_") and len(value) == model.ngeom:
        return value[geoms]
    # BVH storage addresses move when environment geoms are added;
    # mesh vertices/faces and every physical mesh field remain hashed.
    if (
        name.startswith(("jnt_", "dof_", "actuator_", "eq_", "mesh_", "tendon_"))
        and name != "mesh_bvhadr"
    ):
        return value
    return None


def solver_fields(options):
    return [
        (name, getattr(options, name))
        for name in dir(options)
        if name != "timestep"
        and isinstance(getattr(options, name), (np.ndarray, int, float))
    ]


def controller_hash():
    path = "teleopit/sim/g1_o6/holding.py"
    current = Path(path).read_text().replace("\r\n", "\n")
    reviewed = subprocess.check_output(
        ["git", "show", REFERENCE + ":" + path], text=True
    )
    if current != reviewed:
        raise ValueError("campaign requires the unchanged reviewed HoldController")
    return hashlib.sha256(current.encode()).hexdigest()


def assert_preserved(actual, reference):
    for key in ("robot_and_ground_sha256", "solver_sha256", "controller_sha256"):
        if actual[key] != reference[key]:
            raise ValueError("undeclared physical/controller change: " + key)


def verify_reviewed_model(path):
    from teleopit.sim.g1_o6.env import Environment

    env = Environment()
    with np.load(path) as original:
        for name in original.files:
            current = (
                env.data.qpos if name == "reset_qpos" else getattr(env.model, name)
            )
            np.testing.assert_array_equal(current, original[name], err_msg=name)
        return {
            "status": "PASS",
            "identical_compiled_fields_and_reset": len(original.files),
        }
