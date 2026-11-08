"""SI-valued experiment settings; dynamics and postures are provisional."""
from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GraspConfig:
    physics_dt: float = 0.002
    control_dt: float = 0.02
    solver_iterations: int = 100
    solver_tolerance: float = 1e-10
    cross_section: str = 'circular'
    radius: float = 0.012
    width: float = 0.024
    height: float = 0.024
    length: float = 0.30
    rung_enabled: bool = True
    rung_pos: tuple = (0.032, 0.0, 0.608)
    rung_quat: tuple = (0.7071067811865476, 0, 0, 0.7071067811865476)
    palm_pos: tuple = (0, 0, 0.5)  # mounting origin, not palm site
    palm_quat: tuple = (1, 0, 0, 0)
    friction: float = 0.8
    contact_timeconst: float = 0.006
    kp: float = 3.0
    kv: float = 0.02
    effort_limit: float = 0.35  # Nm, provisional, NOT manufacturer torque
    joint_damping: float = 0.002
    armature: float = 0.0002
    command_rate: float = 1.0  # rad/s; URDF velocity field
    actuator_delay_steps: int = 0
    # Hardware command order: thumb pitch, thumb yaw, index, middle, ring, pinky.
    open_pose: tuple = (0, 0, 0, 0, 0, 0)
    closed_pose: tuple = (0.35, 0.95, 1.35, 1.35, 1.35, 1.35)
    residual_bounds: tuple = (0.05, 0.10, 0.12, 0.12, 0.12, 0.12)
    phase_durations: tuple = (0.5, 2.5, 4.0, 2.5)
    fixture_ramp: float = 1.0
    unload_duration: float = 0.5
    evaluation_mode: str = 'unsupported'
    fixture_stiffness: tuple = (1500, 1500, 1500, 8, 8, 8)
    fixture_damping: tuple = (35, 35, 35, 0.3, 0.3, 0.3)
    fixture_force_limit: float = 30.0
    fixture_torque_limit: float = 2.0
    load_force: tuple = (0, 0, -1)
    load_torque: tuple = (0, 0, 0)
    load_point: tuple = (0, 0, 0)  # offset from palm_frame in palm coordinates
    load_frame: str = 'world'
    load_profile: str = 'ramp'
    load_ramp: float = 1.5
    impulse_duration: float = 0.10
    verification_window: float = 0.3
    max_translation: float = 0.015
    max_rotation: float = 0.35
    pullout_distance: float = 0.08
    max_slip_speed: float = 0.02
    max_effort_fraction: float = 0.98
    min_normal_force: float = 0.05
    max_penetration: float = 0.003
    release_clearance: float = 0.002
    velocity_scale: float = 5.0
    distance_scale: float = 0.1
    error_scale: float = 1.0
    linear_velocity_scale: float = 1.0
    angular_velocity_scale: float = 5.0
    observation_noise: float = 0.0  # std in normalized coordinates
    observation_delay_steps: int = 0
    observation_dropout: float = 0.0  # sample hold probability

    @property
    def dimensions(self):
        return (2 * self.radius,) * 2 if self.cross_section == 'circular' else (self.width, self.height)

    def validate(self):
        positive = ('physics_dt', 'control_dt', 'radius', 'width', 'height', 'length',
                    'contact_timeconst', 'kp', 'effort_limit', 'command_rate',
                    'fixture_force_limit', 'fixture_torque_limit', 'fixture_ramp',
                    'unload_duration', 'load_ramp', 'impulse_duration', 'verification_window',
                    'max_translation', 'max_rotation', 'pullout_distance', 'max_slip_speed',
                    'min_normal_force', 'max_penetration', 'release_clearance', 'velocity_scale',
                    'distance_scale', 'error_scale', 'linear_velocity_scale', 'angular_velocity_scale',
                    'solver_tolerance')
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('friction', 'kv', 'joint_damping', 'armature', 'observation_noise'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('actuator_delay_steps', 'observation_delay_steps', 'solver_iterations'):
            value = getattr(self, name)
            if not isinstance(value, int) or value < (1 if name == 'solver_iterations' else 0):
                raise ValueError(f'invalid {name}')
        self._validate_vectors()
        self._validate_choices()
        ratio = self.control_dt / self.physics_dt
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError('control_dt must be an integer multiple of physics_dt')
        if self.fixture_ramp + self.unload_duration >= self.phase_durations[2]:
            raise ValueError('HOLD needs an unsupported interval before unloading')
        if not 0 <= self.observation_dropout <= 1 or not 0 < self.max_effort_fraction <= 1:
            raise ValueError('invalid dropout or effort fraction')

    def _validate_vectors(self):
        sizes = {3: ('rung_pos', 'palm_pos', 'load_force', 'load_torque', 'load_point'),
                 4: ('rung_quat', 'palm_quat', 'phase_durations'),
                 6: ('open_pose', 'closed_pose', 'residual_bounds', 'fixture_stiffness', 'fixture_damping')}
        for size, names in sizes.items():
            for name in names:
                value = np.asarray(getattr(self, name))
                if value.shape != (size,) or not np.isfinite(value).all():
                    raise ValueError(f'{name} must contain {size} finite values')
        for name in ('rung_quat', 'palm_quat'):
            if not np.isclose(np.linalg.norm(getattr(self, name)), 1):
                raise ValueError(f'{name} must be a unit wxyz quaternion')
        for name in ('residual_bounds', 'fixture_stiffness', 'fixture_damping', 'phase_durations'):
            if np.min(getattr(self, name)) < 0:
                raise ValueError(f'{name} cannot be negative')
        if min(self.phase_durations) <= 0:
            raise ValueError('phase durations must be positive')

    def _validate_choices(self):
        choices = {'cross_section': ('circular', 'rectangular'),
                   'evaluation_mode': ('acquisition', 'unsupported'),
                   'load_frame': ('world', 'palm'),
                   'load_profile': ('constant', 'ramp', 'impulse')}
        for name, options in choices.items():
            if getattr(self, name) not in options:
                raise ValueError(f'{name} must be one of {options}')


def load_config(path: str | Path | None) -> GraspConfig:
    values = {} if path is None else json.loads(Path(path).read_text())
    unknown = values.keys() - {f.name for f in fields(GraspConfig)}
    if unknown:
        raise ValueError(f'unknown settings: {sorted(unknown)}')
    config = GraspConfig(**values)
    config.validate()
    return config
