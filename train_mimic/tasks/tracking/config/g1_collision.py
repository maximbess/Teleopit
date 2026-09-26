"""Apply the OmniRetarget collision overlay to canonical G1 bodies.

Only collision geoms and mesh assets change. Kinematics, inertias, visuals,
actuators, and grip/foot sites remain owned by the canonical model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco

from teleopit.runtime.g1_collision_assets import (
    ASSET_DIR, BODY_MAP, REPLACED_GEOMS, REVISION, SCHEMA_VERSION,
    CANONICAL_MESH_OVERRIDES, DONOR_TO_CANONICAL_TRANSLATIONS,
)

# Bit 0 is the climb face and the ground. Robot geoms use bit 1 and listen
# for bit 0, so they hit the ladder and the floor without hitting each other.
ROBOT_CONTYPE = 2
ROBOT_CONAFFINITY = 1


def apply_g1_collision_overlay(spec: mujoco.MjSpec, asset_dir: Path = ASSET_DIR) -> None:
    manifest_path = asset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "OmniRetarget G1 collision assets are missing. Install the collision-build "
            "extra with pip install -e \".[collision-build]\", then run "
            "python scripts/setup/download_assets.py --only g1_collision"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("revision") != REVISION:
        raise ValueError("G1 collision asset version mismatch; rebuild with download_assets.py --only g1_collision")
    for body, relative in CANONICAL_MESH_OVERRIDES.items():
        entry = manifest.get("canonical_sources", {}).get(body, {})
        source = ASSET_DIR.parent / relative
        if entry.get("file") != relative or not source.is_file() or entry.get("sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("Canonical G1 revision mismatch; rebuild collision assets with --only g1_collision")
    bodies = {body.name: body for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY)}
    parts = manifest["parts"]
    if {part["source_body"] for part in parts} != set(BODY_MAP):
        raise ValueError("Incomplete OmniRetarget collision manifest; rebuild collision assets")
    # Validate before mutating the spec. Never silently fall back to primitives.
    for part in parts:
        if BODY_MAP.get(part["source_body"]) != part["body"] or part["body"] not in bodies:
            raise ValueError(f"Collision body mapping mismatch: {part}")
        path = (asset_dir / part["file"]).resolve()
        if not path.is_relative_to(asset_dir.resolve()):
            raise ValueError("Collision mesh path escapes asset directory")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != part["sha256"]:
            raise ValueError(f"Missing or corrupt collision mesh {path}; rebuild collision assets")
    geoms = {geom.name: geom for geom in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)}
    missing = REPLACED_GEOMS - geoms.keys()
    if missing:
        raise ValueError(f"Canonical G1 collision geoms missing: {sorted(missing)}")
    for name in REPLACED_GEOMS:
        spec.delete(geoms[name])
    # The foot shell encloses the distal shin/ankle bearing across the two
    # ankle axes. Keep that pair excluded so the shell and its own shin stay
    # one assembly. Robot-robot contacts are also off via the geom bitmask;
    # ladder contacts stay on.
    for side in ("left", "right"):
        spec.add_exclude(bodyname1=f"{side}_knee_link", bodyname2=f"{side}_ankle_roll_link")
    for index, part in enumerate(parts):
        mesh_name = f"omni_g1_{index:03d}"
        spec.add_mesh(name=mesh_name, file=str((asset_dir / part["file"]).resolve()))
        source = part["source_body"]
        # Foot prefix also supports contact diagnostics and material overrides.
        if source.endswith("ankle_roll_link"):
            prefix = source.split("_")[0] + "_foot"
        else:
            prefix = source.removesuffix("_link")
        bodies[part["body"]].add_geom(
            name=f"{prefix}_omni_{index:03d}_collision",
            type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name,
            density=0.0, contype=ROBOT_CONTYPE, conaffinity=ROBOT_CONAFFINITY, group=3,
            pos=DONOR_TO_CANONICAL_TRANSLATIONS.get(source, (0.0, 0.0, 0.0)),
            condim=3 if source.endswith("ankle_roll_link") else 1,
            rgba=(0.2, 0.7, 0.35, 0.4),
        )
