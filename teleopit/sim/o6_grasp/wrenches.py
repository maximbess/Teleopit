"""World-axis wrenches about palm_frame; force application includes offset moments."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .controller import fixture_gain, load_gain


def bounded(vector, limit):
    return vector * min(1., limit / max(np.linalg.norm(vector), 1e-12))


def fixture_wrench(env, gain):
    cfg, data = env.config, env.data
    palm = data.site('palm_frame')
    error = np.r_[env.fixture_position - palm.xpos,
                  Rotation.from_matrix(env.fixture_rotation @ palm.xmat.reshape(3, 3).T).as_rotvec()]
    velocity = env.palm_velocity(local=False)
    velocity = np.r_[velocity[3:], velocity[:3]]
    wrench = gain * (np.asarray(cfg.fixture_stiffness) * error - np.asarray(cfg.fixture_damping) * velocity)
    return np.r_[bounded(wrench[:3], cfg.fixture_force_limit), bounded(wrench[3:], cfg.fixture_torque_limit)]


def external_wrench(env, gain):
    cfg = env.config
    rotation = env.data.site('palm_frame').xmat.reshape(3, 3)
    force, torque = gain * np.asarray(cfg.load_force), gain * np.asarray(cfg.load_torque)
    if cfg.load_frame == 'palm':
        force, torque = rotation @ force, rotation @ torque
    offset = rotation @ np.asarray(cfg.load_point)
    return np.r_[force, torque + np.cross(offset, force)]


def apply_wrenches(env, mode, elapsed):
    cfg, data = env.config, env.data
    gain = fixture_gain(cfg, mode, elapsed)
    fixture = fixture_wrench(env, gain)
    external = external_wrench(env, load_gain(cfg, mode, elapsed))
    # Rebuild each substep; mj_applyFT adds to its destination.
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    total = fixture + external
    mujoco.mj_applyFT(env.model, data, total[:3], total[3:],
                     data.site('palm_frame').xpos, env.palm_body, data.qfrc_applied)
    return fixture, external, gain
