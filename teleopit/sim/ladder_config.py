"""Full-size ladder dimensions and provenance, resolved into fixed world geometry.

Rung length is lateral Y, depth is approach X, thickness is vertical Z.
Pitch is the center interval; its convention is always explicit. Front is the
negative-X approach face (historical geom prefix ``left_ladder``).
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from functools import lru_cache

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REFERENCE_SOURCE = "Teleopit cc0cd5393c670e7801bb7d242c700e6af4d47303"
LADDERMAN_SOURCE = "amazon-far/LadderMan 9a8d5ed2b44621e84cc40e8efee7abe4b1543025 cfg/task/base/ladder.yaml and active_adaptation/envs/ladder_env.py"
ZARGES_SOURCE = "https://www.zarges.com/de/produkte/r13step-b (80 mm depth; 265 mm vertical interval only)"
REFERENCE_ANGLE = math.degrees(math.atan2(2.8, 1.05))
REFERENCE_PITCH = math.hypot(2.8, 1.05) / 15


@lru_cache(maxsize=1)
def reference_rung_centers():
    """Use the same serialized/compiled transforms as the coupled scene."""
    from .ladder_geometry import add_ladder

    spec = mujoco.MjSpec()
    add_ladder(spec)
    model = mujoco.MjModel.from_xml_string(spec.to_xml())
    return tuple(
        tuple(model.geom(f"left_ladder_rung_{i:02d}").pos) for i in range(1, 15)
    )


@dataclass(frozen=True)
class LadderConfig:
    identity: str = "G0"
    length_m: float = 0.700
    depth_m: float = 0.110
    thickness_m: float = 0.070
    inclination_deg: float = REFERENCE_ANGLE
    pitch_m: float = REFERENCE_PITCH
    pitch_convention: str = "along_rail"
    first_center_m: tuple = (-0.98, 0.0, 2.8 / 15)
    rung_count: int = 14
    rail_style: str = "reference_capsule"
    rail_plane_m: float = 0.100
    rail_lateral_m: float = 0.100
    apex_height_m: float = 2.8
    rear_present: bool = True
    front_collision: bool = True
    rear_collision: bool = False
    top_structure: tuple = ()
    dimension_sources: tuple = (("all", REFERENCE_SOURCE),)

    def __post_init__(self):
        self.validate_dimensions()
        self.validate_structure()

    def validate_dimensions(self):
        positive = (
            self.length_m,
            self.depth_m,
            self.thickness_m,
            self.pitch_m,
            self.rail_plane_m,
            self.rail_lateral_m,
            self.apex_height_m,
        )
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("ladder dimensions must be finite and positive")
        if not 0 < self.inclination_deg < 90:
            raise ValueError("inclination must be between floor and vertical")
        if len(self.first_center_m) != 3 or not np.isfinite(self.first_center_m).all():
            raise ValueError("first rung center must be finite XYZ")

    def validate_structure(self):
        if self.pitch_convention not in ("along_rail", "vertical"):
            raise ValueError("pitch convention must be explicit")
        if type(self.rung_count) is not int or self.rung_count < 1:
            raise ValueError("rung count must be positive")
        if self.rail_style not in (
            "reference_capsule",
            "ladderman_box",
            "commercial_box",
        ):
            raise ValueError("unknown rail construction")
        if self.top_structure:
            raise ValueError("this campaign declares no additional top structure")

    @property
    def pitch_xyz(self):
        angle = math.radians(self.inclination_deg)
        along = self.pitch_m
        if self.pitch_convention == "vertical":
            along /= math.sin(angle)
        return np.array([along * math.cos(angle), 0.0, along * math.sin(angle)])

    def centers(self):
        # Preserve the historical arithmetic exactly for the dimension controls.
        if self.historical_layout:
            return np.array(
                [[-1.05 * (1 - i / 15), 0.0, 2.8 * (i / 15)] for i in range(1, 15)]
            )
        return (
            np.array(self.first_center_m)
            + np.arange(self.rung_count)[:, None] * self.pitch_xyz
        )

    @property
    def historical_layout(self):
        return (
            self.rung_count,
            self.pitch_m,
            self.pitch_convention,
            self.inclination_deg,
            tuple(self.first_center_m),
        ) == (
            14,
            REFERENCE_PITCH,
            "along_rail",
            REFERENCE_ANGLE,
            (-0.98, 0.0, 2.8 / 15),
        )

    @property
    def historical_rails(self):
        return self.historical_layout and (
            self.length_m,
            self.rail_style,
            self.rail_plane_m,
            self.rail_lateral_m,
            self.apex_height_m,
            self.rear_present,
            self.front_collision,
            self.rear_collision,
        ) == (0.7, "reference_capsule", 0.1, 0.1, 2.8, True, True, False)

    def digest(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


def geometry_matrix():
    base = LadderConfig()
    result = {"G0": base}
    for name, depth, thickness in (
        ("G1", 0.110, 0.0381),
        ("G2", 0.080, 0.070),
        ("G3", 0.080, 0.0381),
    ):
        result[name] = replace(
            base,
            identity=name,
            depth_m=depth,
            thickness_m=thickness,
            dimension_sources=(
                ("unchanged dimensions", REFERENCE_SOURCE),
                ("depth_m,thickness_m", "declared controlled geometry experiment"),
            ),
        )
    pitch = 0.25 * np.array([1.05, 0.0, 2.8]) / math.hypot(1.05, 2.8)
    first = base.centers()[3] - 3 * pitch
    result["G4"] = replace(
        base,
        identity="G4",
        pitch_m=0.25,
        first_center_m=tuple(first),
        apex_height_m=float((first + 14 * pitch)[2]),
        dimension_sources=(
            ("unchanged dimensions", REFERENCE_SOURCE),
            (
                "pitch_m,first_center_m,rail endpoints",
                "250 mm along rail; original rung 4 anchored; rails extended to floor and one pitch above last rung",
            ),
        ),
    )
    result["G5"] = LadderConfig(
        identity="G5",
        length_m=0.6,
        depth_m=0.08,
        thickness_m=0.0381,
        inclination_deg=65.5,
        pitch_m=0.25,
        first_center_m=(-0.58, 0.0, 0.19095),
        rung_count=6,
        rail_style="ladderman_box",
        rail_plane_m=0.08 * math.sin(math.radians(65.5)),
        rail_lateral_m=0.0381,
        apex_height_m=0.19095 + 1.25 * math.sin(math.radians(65.5)),
        rear_present=False,
        dimension_sources=(
            ("all geometry dimensions and rail transforms", LADDERMAN_SOURCE),
            (
                "fixed world, contact properties",
                "Teleopit controlled geometry comparison",
            ),
        ),
    )
    angle = math.radians(70)
    result["G6"] = LadderConfig(
        identity="G6",
        length_m=0.54,
        depth_m=0.08,
        thickness_m=0.0381,
        inclination_deg=70,
        pitch_m=0.265,
        pitch_convention="vertical",
        first_center_m=(-(2.35 - 0.265) / math.tan(angle), 0.0, 0.265),
        rung_count=8,
        rail_style="commercial_box",
        rail_plane_m=0.068,
        rail_lateral_m=0.025,
        apex_height_m=2.35,
        rear_collision=True,
        dimension_sources=(
            ("depth_m,pitch_m,pitch_convention", ZARGES_SOURCE),
            (
                "length_m,thickness_m,inclination_deg,first_center_m,rung_count,rail_plane_m,rail_lateral_m,apex_height_m,constant width,no top structure,fixed world",
                "declared simplified commercial stepladder approximation; not a digital twin or certification reconstruction",
            ),
        ),
    )
    return result


def rail_geometry(config, rear=False):
    """Return exact full dimensions and world transforms for both rails."""
    angle = math.radians(config.inclination_deg)
    first, last = config.centers()[[0, -1]]
    if config.rail_style == "ladderman_box":
        length = (config.rung_count - 1) * config.pitch_m
        center = (first + last) / 2
        rotation = Rotation.from_euler("z", -math.pi) * Rotation.from_euler("y", angle)
        dimensions = (length, config.rail_lateral_m, config.rail_plane_m)
        lateral = (config.length_m + config.rail_lateral_m) / 2
    else:
        bottom = np.array([first[0] - first[2] / math.tan(angle), 0.0, 0.0])
        top = bottom + np.array(
            [config.apex_height_m / math.tan(angle), 0.0, config.apex_height_m]
        )
        if config.historical_rails:
            bottom, top = np.array([-1.05, 0.0, 0.0]), np.array([0.0, 0.0, 2.8])
        center = (bottom + top) / 2
        length = float(np.linalg.norm(top - bottom))
        rotation = Rotation.from_euler("y", math.pi / 2 - angle)
        dimensions = (config.rail_plane_m, config.rail_lateral_m, length)
        lateral = config.length_m / 2
        if config.rail_style == "commercial_box":
            lateral += config.rail_lateral_m / 2
    if rear:
        center[0] *= -1
        rotation = Rotation.from_euler("z", math.pi) * rotation
    return [
        (np.array([center[0], sign * lateral, center[2]]), rotation, dimensions)
        for sign in (-1, 1)
    ]


def add_configured_ladder(spec, config):
    from teleopit.sim.ladder_geometry import LADDER_RUNG

    for side in ("left", "right") if config.rear_present else ("left",):
        rear = side == "right"
        collision = config.rear_collision if rear else config.front_collision
        body = spec.worldbody.add_body(name=side + "_ladder_body")
        common = {
            "contype": int(collision),
            "conaffinity": 0,
            "condim": 4,
            "solref": LADDER_RUNG.solref,
            "solimp": LADDER_RUNG.solimp,
            "rgba": (0.55, 0.55, 0.58, 1.0),
        }
        for i, (position, rotation, dimensions) in enumerate(
            rail_geometry(config, rear), 1
        ):
            rail = dict(
                name=f"{side}_ladder_rail_{i:02d}",
                pos=position,
                quat=rotation.as_quat()[[3, 0, 1, 2]],
                friction=(1.4, 0.02, 0.002),
                margin=0.002,
                **common,
            )
            if config.rail_style == "reference_capsule":
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=(config.rail_plane_m / 2, dimensions[2] / 2),
                    **rail,
                )
            else:
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=np.array(dimensions) / 2,
                    **rail,
                )
        for i, center in enumerate(config.centers(), 1):
            if rear:
                center[0] *= -1
            name = f"{side}_ladder_rung_{i:02d}"
            body.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=center,
                size=np.array([config.depth_m, config.length_m, config.thickness_m])
                / 2,
                friction=LADDER_RUNG.friction,
                margin=0.0,
                **common,
            )
            body.add_site(
                name=name + "_grip",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=(
                    *center[:1],
                    -config.length_m / 2,
                    center[2],
                    center[0],
                    config.length_m / 2,
                    center[2],
                ),
                size=(0.075,),
                group=3,
                rgba=(0.15, 0.95, 0.25, 0.25),
            )


def compiled_element(model, data, geom):
    kind = model.geom_type[geom]
    dimensions = 2 * model.geom_size[geom]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius, half_length = model.geom_size[geom, :2]
        dimensions = np.array([2 * radius, 2 * radius, 2 * (half_length + radius)])
        extent = np.abs(rotation[:, 2]) * half_length + radius
    else:
        extent = np.abs(rotation) @ (dimensions / 2)
    center = data.geom_xpos[geom]
    return {
        "position_world_m": center.tolist(),
        "rotation_world": rotation.ravel().tolist(),
        "full_dimensions_xyz_m": dimensions.tolist(),
        "half_sizes_m": model.geom_size[geom].tolist(),
        "type": int(kind),
        "collision_mask": [
            int(model.geom_contype[geom]),
            int(model.geom_conaffinity[geom]),
        ],
        "bounds_xyz_m": [(center - extent).tolist(), (center + extent).tolist()],
    }


def compiled_ladder(model, config):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    elements = compiled_elements(model, data)
    rungs = {k: v for k, v in elements.items() if "_rung_" in k}
    rails = {k: v for k, v in elements.items() if "_rail_" in k}
    centers = [v["position_world_m"] for k, v in rungs.items() if k.startswith("left_")]
    delta = np.diff(centers, axis=0)
    bounds = np.array([v["bounds_xyz_m"] for v in elements.values()])
    pitch = delta[0] if len(delta) else np.zeros(3)
    return {
        "configuration": asdict(config),
        "geometry_sha256": config.digest(),
        "rungs": rungs,
        "rails": rails,
        "top_structure": [],
        "front_pitch": {
            "vertical_m": float(pitch[2]),
            "horizontal_m": float(pitch[0]),
            "along_rail_m": float(np.linalg.norm(pitch)),
        },
        "overall_bounds_xyz_m": [
            bounds[:, 0].min(axis=0).tolist(),
            bounds[:, 1].max(axis=0).tolist(),
        ],
        "surface_displacement_from_G0_m": {
            "front_face_x": (0.110 - config.depth_m) / 2,
            "top_face_z": (config.thickness_m - 0.070) / 2,
        },
    }


def compiled_elements(model, data):
    return {
        model.geom(i).name: compiled_element(model, data, i)
        for i in range(model.ngeom)
        if "_ladder_" in model.geom(i).name
    }
