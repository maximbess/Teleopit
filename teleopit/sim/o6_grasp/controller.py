"""Deterministic sequence; HOLD is a command mode, never a success label."""
from collections import deque
from enum import IntEnum

import numpy as np


class Mode(IntEnum):
    OPEN = 0
    CLOSE = 1
    HOLD = 2
    RELEASE = 3


def smooth(value):
    t = np.clip(value, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def phase(time, durations):
    ends = np.cumsum(durations)
    index = min(int(np.searchsorted(ends, time + 1e-10, side='right')), 3)
    elapsed = max(0., time - (ends[index - 1] if index else 0))
    return Mode(index), elapsed, float(np.clip(elapsed / durations[index], 0, 1))


def nominal_grasp(config, mode, progress):
    """Size-specific reference postures; dimensions are intentionally not extrapolated."""
    amount = {Mode.OPEN: 0., Mode.CLOSE: smooth(progress), Mode.HOLD: 1.,
              Mode.RELEASE: 1 - smooth(progress)}[mode]
    return np.asarray(config.open_pose) + amount * (np.asarray(config.closed_pose) - config.open_pose)


def validate_action(action):
    action = np.asarray(action, dtype=float)
    if action.shape != (6,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError('residual action must be six finite values in [-1, 1]')
    return action


class Controller:
    def __init__(self, config, limits):
        self.config, self.limits = config, limits
        for posture in (config.open_pose, config.closed_pose):
            if np.any(np.asarray(posture) < limits[:, 0]) or np.any(np.asarray(posture) > limits[:, 1]):
                raise ValueError('postures must respect model joint limits')
        self.target = np.array(config.open_pose, dtype=float)
        self.previous_action = np.zeros(6)
        self.release_residual = np.zeros(6)
        self.mode = Mode.OPEN
        self.queue = deque([self.target.copy()] * config.actuator_delay_steps)

    def command(self, time, action):
        cfg = self.config
        mode, elapsed, progress = phase(time, cfg.phase_durations)
        if mode == Mode.RELEASE and self.mode != Mode.RELEASE:
            self.release_residual = self.previous_action.copy()
        residual = np.zeros(6)
        if mode in (Mode.CLOSE, Mode.HOLD):
            residual = action
        elif mode == Mode.RELEASE:
            residual = self.release_residual * (1 - smooth(progress))
        self.mode = mode
        nominal = nominal_grasp(cfg, mode, progress)
        desired = np.clip(nominal + np.asarray(cfg.residual_bounds) * residual,
                          self.limits[:, 0], self.limits[:, 1])
        delta = cfg.command_rate * cfg.control_dt
        self.target += np.clip(desired - self.target, -delta, delta)
        self.previous_action = residual.copy()
        self.queue.append(self.target.copy())
        applied = self.queue.popleft()
        return applied, nominal, mode, elapsed, progress


def fixture_gain(config, mode, elapsed):
    if config.evaluation_mode == 'acquisition' or mode in (Mode.OPEN, Mode.CLOSE):
        return 1.0
    if mode == Mode.HOLD:
        return float(1 - smooth(elapsed / config.fixture_ramp))
    return float(smooth(elapsed / config.fixture_ramp))


def load_gain(config, mode, elapsed):
    if mode != Mode.HOLD or elapsed < config.fixture_ramp:
        return 0.0
    loaded_time = elapsed - config.fixture_ramp
    gain = {'constant': 1.0, 'ramp': smooth(loaded_time / config.load_ramp),
            'impulse': np.sin(np.pi * np.clip(loaded_time / config.impulse_duration, 0, 1)) ** 2}[config.load_profile]
    remaining = config.phase_durations[2] - elapsed
    return float(gain * smooth(remaining / config.unload_duration))
