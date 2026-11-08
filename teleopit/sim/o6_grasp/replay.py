"""Timestamped wrench ON the hand; never infer sign, units or origin from a file."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .config import _check_vector
from .controller import Mode, smooth

SCHEMA = "o6_wrist_load_v1"
UNITS = {"time": "s", "force": "N", "torque": "N*m"}
FRAME_SITES = {"palm": "palm_frame", "rung": "rung_frame"}


def _metadata(document):
    expected = {
        "schema",
        "source",
        "units",
        "frame",
        "application_site",
        "point_m",
        "samples",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError(
            f"wrist replay requires exactly these fields: {sorted(expected)}"
        )
    if document["schema"] != SCHEMA or document["units"] != UNITS:
        raise ValueError(f"wrist replay must use {SCHEMA} and SI units {UNITS}")
    if document["frame"] not in ("world", "palm", "rung"):
        raise ValueError("replay frame must be world, palm or rung")
    if document["application_site"] not in ("wrist_mount", "palm_frame"):
        raise ValueError("application_site must be wrist_mount or palm_frame")
    _validate_source(document["source"])
    return _check_vector(document["point_m"], 3, "point_m")


def _validate_source(source):
    if not isinstance(source, dict) or source.get("kind") not in (
        "synthetic",
        "simulation",
        "hardware",
    ):
        raise ValueError("source.kind must be synthetic, simulation or hardware")
    if (
        not isinstance(source.get("description"), str)
        or not source["description"].strip()
    ):
        raise ValueError(
            "source.description must identify the recording or synthetic generator"
        )


def _samples(document):
    samples = document["samples"]
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("replay needs at least two samples")
    times, wrenches = [], []
    for row in samples:
        if not isinstance(row, dict) or set(row) != {"time_s", "force_N", "torque_Nm"}:
            raise ValueError("each sample needs time_s, force_N and torque_Nm")
        times.append(float(row["time_s"]))
        wrenches.append(
            np.r_[
                _check_vector(row["force_N"], 3, "force_N"),
                _check_vector(row["torque_Nm"], 3, "torque_Nm"),
            ]
        )
    times = np.asarray(times)
    if not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError(
            "timestamps must be finite, start at zero and strictly increase"
        )
    return times, np.asarray(wrenches)


@dataclass(frozen=True)
class WristLoadReplay:
    path: str
    raw_bytes: bytes
    times: np.ndarray
    wrenches: np.ndarray
    frame: str
    application_site: str
    point_m: np.ndarray

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        document = json.loads(raw)
        point = _metadata(document)
        times, wrenches = _samples(document)
        for value in (times, wrenches, point):
            value.setflags(write=False)
        return cls(
            str(path),
            raw,
            times,
            wrenches,
            document["frame"],
            document["application_site"],
            point,
        )

    @property
    def duration(self):
        return float(self.times[-1])

    @property
    def metadata(self):
        document = json.loads(self.raw_bytes)
        return {
            "schema": SCHEMA,
            "path": self.path,
            "sha256": hashlib.sha256(self.raw_bytes).hexdigest(),
            "source": document["source"],
            "frame": self.frame,
            "application_site": self.application_site,
            "point_m": self.point_m.tolist(),
            "duration_s": self.duration,
            "sample_count": len(self.times),
        }

    def sample(self, time):
        """Piecewise linear, no looping/holding/extrapolation outside the trace."""
        if time < 0 or time > self.duration:
            return np.zeros(6)
        return np.array(
            [np.interp(time, self.times, column) for column in self.wrenches.T]
        )

    def world_wrench(self, data, wrench):
        """Shift the site's application-point moment to the current palm origin."""
        rotation = (
            np.eye(3)
            if self.frame == "world"
            else data.site(FRAME_SITES[self.frame]).xmat.reshape(3, 3)
        )
        force, moment = rotation @ wrench[:3], rotation @ wrench[3:]
        site = data.site(self.application_site)
        point = site.xpos + site.xmat.reshape(3, 3) @ self.point_m
        offset = point - data.site("palm_frame").xpos
        return np.r_[force, moment + np.cross(offset, force)]


def with_replay(config, path, *, fade_duration=0.1, scale=1.0):
    """Configure a trace; the environment reads and owns the snapshot once."""
    return replace(
        config,
        load_profile="replay",
        wrist_load_path=str(Path(path).resolve()),
        replay_fade_duration=fade_duration,
        replay_scale=scale,
        load_force=(0, 0, 0),
        load_torque=(0, 0, 0),
    )


def fit_replay(config, replay):
    if config.replay_fade_duration * 2 > replay.duration:
        raise ValueError("replay fades must fit within the trace duration")
    # Reserve a full unload interval before RELEASE/catch; never truncate a trace.
    minimum = config.fixture_ramp + replay.duration + config.unload_duration
    hold = max(
        config.phase_durations[2],
        np.ceil(minimum / config.control_dt - 1e-10) * config.control_dt,
    )
    durations = (*config.phase_durations[:2], float(hold), config.phase_durations[3])
    result = replace(config, phase_durations=durations)
    result.validate()
    return result


def replay_wrench(env, mode, elapsed):
    replay, cfg = env.replay, env.config
    time = elapsed - cfg.fixture_ramp
    active = mode == Mode.HOLD and 0 <= time <= replay.duration
    requested = replay.sample(time) if active else np.zeros(6)
    envelope = float(active)
    if active and cfg.replay_fade_duration > 0:
        envelope *= smooth(time / cfg.replay_fade_duration) * smooth(
            (replay.duration - time) / cfg.replay_fade_duration
        )
    applied = cfg.replay_scale * envelope * requested
    env.last_replay = {
        "active": bool(active),
        "time_s": float(time) if mode == Mode.HOLD else None,
        "envelope": float(envelope),
        "requested_wrench": requested.tolist(),
        "scaled_wrench": applied.tolist(),
    }
    return replay.world_wrench(env.data, applied)
