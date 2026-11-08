"""The climbing rung must support the hand without a wrist fixture or extra load."""

from dataclasses import replace

import numpy as np
import pytest

from teleopit.sim.o6_grasp import O6GraspEnv
from teleopit.sim.o6_grasp.evaluate import summarize
from teleopit.sim.o6_grasp.presets import ladder_rung_config


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"phase_durations": (0.5, 2.5, 32.0, 2.5)},
        {
            "phase_durations": (0.5, 2.5, 32.0, 2.5),
            "physics_dt": 0.0005,
            "solver_iterations": 400,
            "solver_tolerance": 1e-12,
        },
    ],
    ids=["nominal", "long_hold", "fine_physics"],
)
def test_ladder_preset_holds_gravity_and_releases(overrides):
    cfg = replace(ladder_rung_config(), **overrides)
    env = O6GraspEnv(cfg)
    _, initial = env.reset(seed=0)
    rows = []
    while True:
        _, _, _, done, info = env.step(np.zeros(6))
        rows.append(info)
        if done:
            break
    result = summarize("gravity", cfg, initial, rows)
    assert result["unsupported_success"], result
    assert result["release_clear"]
    held = [r for r in rows if r["unsupported"] and r["time"] > 6]
    assert held and all(r["verification"] == "VERIFIED" for r in held)
    for row in held:
        np.testing.assert_array_equal(row["fixture_wrench"], np.zeros(6))
        np.testing.assert_array_equal(row["external_wrench"], np.zeros(6))
    loaded = [c for c in held[-1]["contacts"] if c["normal_force"] >= 0.05]
    assert any("thumb" in c["body"] for c in loaded)
    normals = np.array([c["normal_world"] for c in loaded])
    assert np.min(normals @ normals.T) < -0.5  # opposing rung surfaces
    np.testing.assert_allclose(
        np.mean([r["contact_wrench"][2] for r in held]),
        np.sum(env.model.body_mass) * 9.81,
        atol=0.12,
    )
