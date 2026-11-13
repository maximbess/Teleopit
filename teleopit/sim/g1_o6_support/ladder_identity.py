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
        if name.startswith("body_") and len(value) == model.nbody:
            selected.append((name, value[bodies]))
        elif name.startswith("geom_") and len(value) == model.ngeom:
            selected.append((name, value[geoms]))
        elif name.startswith(("jnt_", "dof_", "actuator_", "eq_", "mesh_", "tendon_")) and name != "mesh_bvhadr":
            # BVH storage addresses move when environment geoms are added;
            # mesh vertices/faces and every physical mesh field remain hashed.
            selected.append((name, value))
    solver = [(name, getattr(model.opt, name)) for name in dir(model.opt)
              if name != "timestep" and isinstance(getattr(model.opt, name), (np.ndarray, int, float))]
    return dict(robot_and_ground_sha256=array_digest(selected), solver_sha256=array_digest(solver),
                controller_sha256=controller_hash(), physics_dt=float(model.opt.timestep))


def controller_hash():
    path = "teleopit/sim/g1_o6/holding.py"
    current = Path(path).read_text().replace("\r\n", "\n")
    reviewed = subprocess.check_output(["git", "show", REFERENCE + ":" + path], text=True)
    if current != reviewed:
        raise ValueError("campaign requires the unchanged reviewed HoldController")
    return hashlib.sha256(current.encode()).hexdigest()


def assert_preserved(actual, reference):
    for key in ("robot_and_ground_sha256", "solver_sha256", "controller_sha256"):
        if actual[key] != reference[key]:
            raise ValueError("undeclared physical/controller change: " + key)
