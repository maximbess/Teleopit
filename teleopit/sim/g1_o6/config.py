"""Predeclared thresholds; never relax them to turn a failed trial green."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    physics_dt: float = 0.001
    control_dt: float = 0.02
    hold_seconds: float = 10.0
    acquire_seconds: float = 3.0
    remove_seconds: float = 2.0
    friction_scale: float = 1.0
    delay_steps: int = 0
    root_offset_x: float = 0.0
    root_offset_z: float = 0.0
    wrist_offset_x: float = 0.0
    disturbance_N: float = 0.0
    max_drift_m: float = 0.03
    max_angle_rad: float = 0.25
    max_slip_m_s: float = 0.02
    max_penetration_m: float = 0.003
    max_joint_error_rad: float = 0.02
    max_mimic_error_rad: float = 0.015
    max_effort_fraction: float = 0.98
    stability_seconds: float = 0.5
    assistance_force_N: float = 600.0
    assistance_moment_Nm: float = 150.0

    def validate(self):
        self.validate_values()
        self.validate_timing()
        if not 0 < self.max_effort_fraction <= 1:
            raise ValueError("invalid effort fraction")
        if min(self.assistance_force_N, self.assistance_moment_Nm) < 0:
            raise ValueError("assistance limits cannot be negative")

    def validate_values(self):
        for key, value in vars(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
        positive = (
            "physics_dt",
            "control_dt",
            "hold_seconds",
            "acquire_seconds",
            "remove_seconds",
            "friction_scale",
            "max_drift_m",
            "max_angle_rad",
            "max_slip_m_s",
            "max_penetration_m",
            "max_joint_error_rad",
            "max_mimic_error_rad",
            "stability_seconds",
        )
        if any(getattr(self, k) <= 0 for k in positive):
            raise ValueError("timing, friction and thresholds must be positive")

    def validate_timing(self):
        ratio = self.control_dt / self.physics_dt
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if not isinstance(self.delay_steps, int) or self.delay_steps < 0:
            raise ValueError("delay_steps must be a nonnegative integer")
