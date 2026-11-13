"""Shared climbing-rung geometry/material, independent of training dependencies."""

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class LadderRung:
    length: float = 0.700
    depth: float = 0.110
    height: float = 0.070
    friction: tuple = (1.8, 0.02, 0.002)
    solref: tuple = (0.005, 1.0)
    solimp: tuple = (0.99, 0.999, 0.001, 0.5, 2.0)

    @property
    def half_size_xyz(self):
        """Training rung: length along world Y, depth X, height Z."""
        return (self.depth / 2, self.length / 2, self.height / 2)


LADDER_RUNG = LadderRung()


_LADDER_NUM_RUNGS = 14
_LADDER_HALF_BASE = 1.05
_LADDER_HEIGHT = 2.80
_LADDER_HALF_WIDTH = LADDER_RUNG.length / 2
_LADDER_RAIL_RADIUS = 0.050
_LADDER_RUNG_HALF_DEPTH = LADDER_RUNG.depth / 2
_LADDER_RUNG_HALF_HEIGHT = LADDER_RUNG.height / 2
_LADDER_GRIP_RADIUS = 0.075

_LADDER_CONTYPE = 1
_LADDER_CONAFFINITY = 0
_LADDER_RAIL_MARGIN = 0.002
_LADDER_BOX_MARGIN = 0.0


def add_ladder(spec, config=None):
    if config is not None:
        from .ladder_config import LadderConfig, add_configured_ladder

        if not config.historical_rails:
            return add_configured_ladder(spec, config)
        # The dimension controls use the historical construction byte for byte,
        # then change only box sizes. G0 retains every compiled numeric field.
        add_reference_ladder(spec)
        if config != LadderConfig():
            for geom in spec.geoms:
                if "_ladder_rung_" in geom.name:
                    geom.size[:] = (
                        config.depth_m / 2,
                        config.length_m / 2,
                        config.thickness_m / 2,
                    )
        return
    add_reference_ladder(spec)


def add_reference_ladder(spec):
    world = spec.worldbody
    # The robot stands on the negative-x face. The positive-x face is the back
    # of the A-frame: keep it for rendering, and leave it out of collision.
    for side, base_x in (("left", -_LADDER_HALF_BASE), ("right", _LADDER_HALF_BASE)):
        collides = side == "left"
        side_body = world.add_body(name=f"{side}_ladder_body")
        for rail_index, y in enumerate(
            (-_LADDER_HALF_WIDTH, _LADDER_HALF_WIDTH),
            start=1,
        ):
            side_body.add_geom(
                name=f"{side}_ladder_rail_{rail_index:02d}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=(base_x, y, 0.0, 0.0, y, _LADDER_HEIGHT),
                size=(_LADDER_RAIL_RADIUS,),
                contype=_LADDER_CONTYPE if collides else 0,
                conaffinity=_LADDER_CONAFFINITY,
                condim=4,
                friction=(1.4, 0.02, 0.002),
                solref=(0.005, 1.0),
                solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
                margin=_LADDER_RAIL_MARGIN,
                rgba=(0.55, 0.55, 0.58, 1.0),
            )

        for rung_index in range(1, _LADDER_NUM_RUNGS + 1):
            fraction = rung_index / (_LADDER_NUM_RUNGS + 1)
            x = base_x * (1.0 - fraction)
            z = _LADDER_HEIGHT * fraction
            fromto = (
                x,
                -_LADDER_HALF_WIDTH,
                z,
                x,
                _LADDER_HALF_WIDTH,
                z,
            )
            rung_name = f"{side}_ladder_rung_{rung_index:02d}"
            side_body.add_geom(
                name=rung_name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(x, 0.0, z),
                size=(
                    _LADDER_RUNG_HALF_DEPTH,
                    _LADDER_HALF_WIDTH,
                    _LADDER_RUNG_HALF_HEIGHT,
                ),
                contype=_LADDER_CONTYPE if collides else 0,
                conaffinity=_LADDER_CONAFFINITY,
                condim=4,
                friction=LADDER_RUNG.friction,
                solref=LADDER_RUNG.solref,
                solimp=LADDER_RUNG.solimp,
                margin=_LADDER_BOX_MARGIN,
                rgba=(0.55, 0.55, 0.58, 1.0),
            )
            side_body.add_site(
                name=f"{rung_name}_grip",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=fromto,
                size=(_LADDER_GRIP_RADIUS,),
                group=3,
                rgba=(0.15, 0.95, 0.25, 0.25),
            )
