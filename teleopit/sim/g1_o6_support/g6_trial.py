"""G6 candidates share the original controller and strict HOLD implementation."""

from pathlib import Path

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.ladder_config import geometry_matrix

from .g6_candidates import G6Candidate
from .g6_observer import G6Observer
from .g6_reach import next_tread_reach
from .ladder_identity import assert_preserved, physical_identity
from .long_hold import run_hold
from .pose_trial import reject_reset, reset_report


def make_environment(candidate, seconds, dt):
    return Environment(
        Config(hold_seconds=seconds, physics_dt=dt, **candidate.reset),
        pose=candidate.pose,
        ladder=geometry_matrix()["G6"],
    )


def run_candidate(payload, output, seconds, dt):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    candidate = G6Candidate.from_dict(payload["candidate"])
    env = make_environment(candidate, seconds, dt)
    assert_preserved(physical_identity(env.model), payload["reference_physics"])
    extra = {
        "g6_campaign": payload,
        "reset_inspection": reset_report(env),
        "next_tread_reach": next_tread_reach(env),
    }
    reasons = list(env.initial_errors)
    if extra["reset_inspection"]["minimum_joint_margin_rad"] < -1e-9:
        reasons.append("initial_joint_range")
    if reasons:
        return reject_reset(env, output, extra, reasons)
    return run_hold(
        output,
        seconds=seconds,
        physics_dt=dt,
        environment=env,
        observer_factory=G6Observer,
        metadata_extra=extra,
    )
