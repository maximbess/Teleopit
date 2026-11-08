"""Reproducible scene presets. Ladder geometry is shared with the climbing task."""

from dataclasses import replace

from teleopit.sim.ladder_geometry import LADDER_RUNG

from .config import GraspConfig


def ladder_rung_config():
    """Test the near upper edge; this posture is not calibrated to the large rung."""
    return replace(
        GraspConfig(),
        cross_section="rectangular",
        length=LADDER_RUNG.length,
        width=LADDER_RUNG.depth,
        height=LADDER_RUNG.height,
        rung_friction=LADDER_RUNG.friction,
        rung_solref=LADDER_RUNG.solref,
        rung_solimp=LADDER_RUNG.solimp,
        # Preserve the world axes of the climbing task, but translate the rung
        # to the standalone bench. Local X remains the longitudinal world Y.
        rung_pos=(LADDER_RUNG.depth / 2 + 0.025, 0.0, 0.608),
        physics_dt=0.001,
        solver_iterations=200,
        solver_tolerance=1e-11,
        load_force=(0, 0, 0),
        load_torque=(0, 0, 0),
    )
