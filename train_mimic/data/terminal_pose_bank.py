"""Select real terminal states without averaging configurations or quaternions."""
from __future__ import annotations

import numpy as np


def pose_distances(positions, quaternions, joints):
    """RMS metric: 3 cm body translation, 0.15 rad rotation, 0.2 rad joint change."""
    positions = np.asarray(positions, dtype=float)[:, 4:6]  # pelvis, torso
    quaternions = np.asarray(quaternions, dtype=float)[:, 4:6].copy()
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    joints = np.asarray(joints, dtype=float)
    translation = np.mean(((positions[:, None]-positions[None, :])/.03)**2, axis=(2, 3))
    dot = np.abs(np.sum(quaternions[:, None]*quaternions[None, :], axis=-1)).clip(0, 1)
    rotation = np.mean((2*np.arccos(dot)/.15)**2, axis=2)
    articulation = np.mean(((joints[:, None]-joints[None, :])/.2)**2, axis=2)
    return np.sqrt(translation+rotation+articulation)


def representatives(distances, count):
    """A medoid first, followed by farthest-point coverage of observed states."""
    d = np.asarray(distances)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or not len(d) or not np.isfinite(d).all():
        raise ValueError('Expected a finite nonempty square distance matrix')
    if not 1 <= count <= len(d):
        raise ValueError('Representative count must be between 1 and bank size')
    selected = [int(np.argmin(d.mean(axis=1)))]
    while len(selected) < count:
        nearest = d[:, selected].min(axis=1)
        nearest[selected] = -np.inf
        selected.append(int(np.argmax(nearest)))
    return selected


def stable_history(frames, required_frames):
    """Reject short histories, episode boundaries, and any unstable frame."""
    if len(frames) < required_frames:
        return False
    frames = list(frames)[-required_frames:]
    return (len({int(f['episode_id']) for f in frames}) == 1
            and all(bool(f['stable']) for f in frames)
            and all(np.isfinite(f[k]).all() for f in frames
                    for k in ('qpos', 'qvel', 'track_pos', 'track_quat')))
