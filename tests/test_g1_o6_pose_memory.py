import json

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.diagnostics import snapshot
from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.posture_report import annotate
from teleopit.sim.g1_o6.run import summarize


def test_streamed_diagnostics_need_not_be_retained_to_preserve_strict_results():
    from teleopit.sim.g1_o6_support.long_hold import summary_row

    env = Environment(
        Config(acquire_seconds=0.02, remove_seconds=0.02, hold_seconds=0.02)
    )
    rows = [annotate(env, snapshot(env))]
    for _ in range(8):
        if env.phase in ("ABORT", "HOLD_COMPLETE"):
            break
        rows.append(annotate(env, env.step()))
    for row in rows:
        row.update(
            diagnostic_v1={"unscored_detailed_trace": list(range(10000))},
            pose_search={"raw": list(range(10000))},
        )
    compact = [summary_row(row) for row in rows]
    assert summarize(env, compact) == summarize(env, rows)
    assert len(json.dumps(compact)) < len(json.dumps(rows)) / 2


def test_near_identical_ik_pose_is_not_an_independent_discovery():
    import numpy as np

    from teleopit.sim.g1_o6_support.pose_trial import duplicate_reset

    reference = {"id": "original", "qpos": [0, 1, 2], "finger_identity": "same"}
    assert duplicate_reset(np.array([0, 1.0001, 2]), "same", [reference]) == "original"
    assert duplicate_reset(np.array([0, 1.01, 2]), "same", [reference]) is None
    assert duplicate_reset(np.array([0, 1, 2]), "changed", [reference]) is None
