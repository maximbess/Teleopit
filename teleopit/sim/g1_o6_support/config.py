"""Predeclared experimental criteria; original physical gates live in Config."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DriftCriteria:
    window_seconds: float = 30.0
    translation_slope_m_s: float = 0.00005
    translation_excursion_m: float = 0.002
    rotation_slope_rad_s: float = 0.0005
    rotation_excursion_rad: float = 0.01


@dataclass(frozen=True)
class TransferCriteria:
    share_tolerance: float = 0.01
    minimum_per_hand_share: float = 0.005
    maximum_per_hand_share: float = 0.15
    offset_bound_rad: float = 0.08
    offset_rate_rad_s: float = 0.005
    ramp_s: float = 10.0
    settle_timeout_s: float = 15.0
    plateau_s: float = 10.0
    recovery_s: float = 10.0
    persistence_s: float = 0.5
    reference_s: float = 2.0
    unload_floor_N: float = 5.0
    unload_reduction: float = 0.8
    wrist_force_N: float = 8.0
    wrist_moment_Nm: float = 0.4
    reaction_tolerance_N: float = 2.0
    unloaded_hold_s: float = 2.0
    clearance_m: float = 0.005
    opening_s: float = 3.0
    trajectory_s: float = 5.0
    retraction_m: float = 0.02
    open_tolerance_rad: float = 0.1
    minimum_open_motion_rad: float = 0.1
    support_motion_m: float = 0.002
