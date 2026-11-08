"""Reproducible scene presets. Ladder geometry is shared with the climbing task."""

from dataclasses import replace

from teleopit.sim.ladder_geometry import LADDER_RUNG

from .config import GraspConfig


def ladder_rung_config():
    """Wrap the upper edge from a tilted wrist pose, with finite servo preload."""
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
        # Mount origin and wxyz orientation (roughly roll -26.4, pitch -16.8,
        # yaw +2.45 degrees). This lets the fingers wrap over the upper edge
        # while the thumb opposes below, without initial interpenetration.
        palm_pos=(0.016, 0.0, 0.535),
        palm_quat=(0.96360256, -0.22279641, -0.14719825, -0.01282755),
        # Thumb pitch/yaw, then index/middle/ring/pinky (radians). Targets
        # near the loaded joint angles avoid the old six-servo saturation.
        closed_pose=(0.05, 0.14, 0.77, 0.98, 1.04, 1.07),
        physics_dt=0.001,
        solver_iterations=200,
        solver_tolerance=1e-11,
        load_force=(0, 0, 0),
        load_torque=(0, 0, 0),
    )
