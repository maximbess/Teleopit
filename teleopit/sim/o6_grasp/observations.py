"""o6_actor_v1: 48 fields; simulated pose/velocity need hardware estimation."""
from collections import deque
from dataclasses import dataclass

import numpy as np

VERSION = 'o6_actor_v1'


@dataclass(frozen=True)
class Field:
    name: str
    size: int
    estimated_on_hardware: bool = False


SPEC = (Field('active_position', 6), Field('active_velocity', 6),
        Field('nominal_error', 6), Field('previous_residual', 6),
        Field('rung_translation_in_palm', 3, True), Field('rung_rotation_columns', 6, True),
        Field('palm_linear_velocity', 3, True), Field('palm_angular_velocity', 3, True),
        Field('cross_section_dimensions', 2), Field('cross_section_type', 2),
        Field('controller_mode', 4), Field('phase_progress', 1))


def build_observation(env, nominal, mode, progress):
    cfg, data = env.config, env.data
    palm, rung = data.site('palm_frame'), data.site('rung_frame')
    rotation = palm.xmat.reshape(3, 3)
    relative = rotation.T @ rung.xmat.reshape(3, 3)
    velocity = env.palm_velocity(local=True)  # MuJoCo ordering: angular, linear
    q = data.qpos[env.active_qpos]
    raw = np.concatenate((q, data.qvel[env.active_dofs], nominal - q,
                          env.controller.previous_action,
                          rotation.T @ (rung.xpos - palm.xpos),
                          relative[:, :2].flatten(order='F'), velocity[3:], velocity[:3],
                          cfg.dimensions, [cfg.cross_section == 'circular', cfg.cross_section == 'rectangular'],
                          np.eye(4)[mode], [progress])).astype(float)
    normalized = raw.copy()
    low, high = env.limits.T
    normalized[:6] = 2 * (q - low) / (high - low) - 1
    normalized[6:12] /= cfg.velocity_scale
    normalized[12:18] /= cfg.error_scale
    normalized[24:27] /= cfg.distance_scale
    normalized[33:36] /= cfg.linear_velocity_scale
    normalized[36:39] /= cfg.angular_velocity_scale
    normalized[39:41] /= cfg.distance_scale
    return raw, normalized


class ObservationSensor:
    """Normalized Gaussian noise, whole-sample delay, then sample-and-hold dropout."""
    def __init__(self, config, seed):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.queue = None
        self.last = None

    def sample(self, observation):
        cfg = self.config
        noisy = np.array(observation, dtype=float)
        indices = np.r_[0:18, 24:39]
        noisy[indices] += self.rng.normal(0, cfg.observation_noise, len(indices))
        if self.queue is None:
            self.queue = deque([noisy.copy()] * cfg.observation_delay_steps)
            self.last = noisy.copy()
        self.queue.append(noisy)
        delayed = self.queue.popleft()
        if self.rng.random() >= cfg.observation_dropout:
            self.last = delayed
        return self.last.astype(np.float32).copy()


class History:
    """Ten observations, oldest first, with current observation in the last row."""
    def __init__(self, env):
        self.env = env
        self.frames = deque(maxlen=10)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        self.frames.extend(obs.copy() for _ in range(10))
        return np.stack(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(self.frames), reward, terminated, truncated, info
