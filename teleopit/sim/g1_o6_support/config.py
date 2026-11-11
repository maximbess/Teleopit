"""Predeclared experimental criteria; original physical gates live in Config."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DriftCriteria:
    window_seconds: float = 30.0
    translation_slope_m_s: float = 0.00005
    translation_excursion_m: float = 0.002
    rotation_slope_rad_s: float = 0.0005
    rotation_excursion_rad: float = 0.01
