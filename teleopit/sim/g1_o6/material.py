"""Selective effective tread friction with unchanged dynamic contact parameters."""

import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

import numpy as np

from .diagnostics import region


@dataclass(frozen=True)
class Material:
    scope: str
    mu: float

    def __post_init__(self):
        if self.scope not in ("H", "M"):
            raise ValueError("material scope must be H (hands) or M (whole tread)")
        if not math.isfinite(self.mu) or self.mu <= 0:
            raise ValueError(
                "effective tangential friction must be finite and positive"
            )


def eligible_pair(model, first, second):
    a, b = sorted(int(model.geom_bodyid[g]) for g in (first, second))
    masks = (model.geom_contype[first] & model.geom_conaffinity[second]) or (
        model.geom_contype[second] & model.geom_conaffinity[first]
    )
    return bool(
        masks
        and model.body_weldid[a] != model.body_weldid[b]
        and a * 65536 + b not in model.exclude_signature
    )


def selected_pairs(model, material):
    rungs = [g for g in range(model.ngeom) if "_ladder_rung_" in model.geom(g).name]
    robots = selected_robot_geoms(model, material.scope)
    return [
        (robot, rung)
        for robot in robots
        for rung in rungs
        if eligible_pair(model, robot, rung)
    ]


def selected_robot_geoms(model, scope):
    robots = [
        g
        for g in range(model.ngeom)
        if region(model, int(model.geom_bodyid[g])) != "environment"
    ]
    if scope == "H":
        robots = [g for g in robots if "_o6_" in model.body(model.geom_bodyid[g]).name]
    return robots


def mix_solver(model, a, b):
    """MuJoCo's dynamic mixing, with direct-format solref handled separately."""
    weights = model.geom_solmix[[a, b]]
    total = float(weights.sum())
    weight = float(weights[0] / total) if total > 1e-15 else 0.5
    solref = weight * model.geom_solref[a] + (1 - weight) * model.geom_solref[b]
    if min(model.geom_solref[a, 0], model.geom_solref[b, 0]) <= 0:
        solref = np.minimum(model.geom_solref[a], model.geom_solref[b])
    solimp = weight * model.geom_solimp[a] + (1 - weight) * model.geom_solimp[b]
    return solref, solimp


def mixed_parameters(model, a, b):
    if model.geom_priority[a] != model.geom_priority[b]:
        winner = a if model.geom_priority[a] > model.geom_priority[b] else b
        friction = model.geom_friction[winner]
        condim = model.geom_condim[winner]
        solref, solimp = model.geom_solref[winner], model.geom_solimp[winner]
    else:
        friction = np.maximum(model.geom_friction[a], model.geom_friction[b])
        condim = max(model.geom_condim[a], model.geom_condim[b])
        solref, solimp = mix_solver(model, a, b)
    return {
        "condim": int(condim),
        "friction": friction[[0, 0, 1, 2, 2]].tolist(),
        "solref": solref.tolist(),
        "solimp": solimp.tolist(),
        "solreffriction": [0.0, 0.0],
        "margin": float(model.geom_margin[a] + model.geom_margin[b]),
        "gap": float(model.geom_gap[a] + model.geom_gap[b]),
    }


def xml_values(parameters):
    return {
        k: " ".join(map(str, v)) if isinstance(v, list) else str(v)
        for k, v in parameters.items()
    }


def configure_material(root, model, material):
    """Pairs are symmetric and generated only from eligible robot/tread masks.

    If the requested coefficient already equals the dynamic mixture, retain
    native contacts. This preserves reference contact ordering and trajectory.
    """
    if material is None:
        return None
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    existing = {
        frozenset((int(a), int(b))) for a, b in zip(model.pair_geom1, model.pair_geom2)
    }
    rows = []
    for a, b in selected_pairs(model, material):
        if frozenset((a, b)) in existing:
            raise ValueError("selected contact already has an explicit pair")
        parameters = mixed_parameters(model, a, b)
        native = parameters["friction"][:2] == [material.mu, material.mu]
        parameters["friction"][:2] = [material.mu, material.mu]
        robot, rung = model.geom(a).name, model.geom(b).name
        if not native:
            ET.SubElement(
                contact,
                "pair",
                name=f"material_{a}_{b}",
                geom1=robot,
                geom2=rung,
                **xml_values(parameters),
            )
        rows.append(
            dict(robot=robot, rung=rung, native_equivalent=native, **parameters)
        )
    return dict(
        **asdict(material),
        requested_tangential_mu=material.mu,
        pairs=rows,
        interpretation="sensitivity level, not a calibrated named material",
    )
