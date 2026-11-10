"""Diagnostic-only schedules and guards, separate from strict grasp thresholds."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiagnosticConfig:
    continuation_s: float = 0.0
    fall_drop_m: float = 0.25
    escape_m: float = 0.5
    emergency_penetration_m: float = 0.01
    free_schedule_s: tuple = (1.0, 3.0, 2.0, 3.0, 1.0)
    acquire_open_s: float = 1.0
    acquire_close_s: float = 3.0
    fixture_remove_s: float = 2.0
    zero_fixture_settle_s: float = 2.0
    load_ramp_s: float = 2.0
    plateau_s: float = 10.0
    settling_position_rad: float = 0.03
    settling_velocity_rad_s: float = 0.05
    persistence_s: float = 0.5
    max_translation_m: float = 0.015
    max_rotation_rad: float = 0.35
    max_slip_m_s: float = 0.02
    max_penetration_m: float = 0.003
    max_joint_error_rad: float = 0.02
    max_mimic_error_rad: float = 0.015
    max_effort_fraction: float = 0.98
    min_normal_N: float = 0.05
    scales: tuple = (0.25, 0.5, 0.75, 1.0)

    def validate(self):
        if not 0 <= self.continuation_s <= 3:
            raise ValueError("continuation must be between 0 and 3 seconds")
        if any(not np.isfinite(v).all() for v in vars(self).values()):
            raise ValueError("diagnostic configuration must be finite")
        if any(np.min(v) <= 0 for k, v in vars(self).items() if k != "continuation_s"):
            raise ValueError("diagnostic thresholds and schedules must be positive")


def emergency(model, data, config, reference, previous_warnings):
    return state_guard(data, config, previous_warnings) or position_guard(
        data.body(reference[0]).xpos, reference[1], config
    )


def state_guard(data, config, previous_warnings):
    if not np.isfinite(np.r_[data.qpos, data.qvel, data.qacc]).all():
        return "nonfinite_state"
    if any(w.number > old for w, old in zip(data.warning, previous_warnings)):
        return "new_numerical_warning"
    if max((-c.dist for c in data.contact), default=0) > config.emergency_penetration_m:
        return "excessive_penetration"
    return None


def position_guard(position, reference, config):
    if position[2] < reference[2] - config.fall_drop_m:
        return "fall_drop"
    if np.linalg.norm(position - reference) > config.escape_m:
        return "escape"
    return None
