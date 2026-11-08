"""Simulation entry points; standalone experiments do not load humanoid runtime."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loop import SimulationLoop

__all__ = ["SimulationLoop"]


def __getattr__(name):
    if name == "SimulationLoop":
        from .loop import SimulationLoop

        return SimulationLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
