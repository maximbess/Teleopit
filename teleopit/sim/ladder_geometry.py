"""Shared climbing-rung geometry/material, independent of training dependencies."""

from dataclasses import dataclass


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
