"""Small real coupled run checks streaming evidence, not physical qualification."""

import gzip
import json


def test_runner_streams_actual_forces_contacts_and_hold_entry(tmp_path, monkeypatch):
    from dataclasses import replace

    from teleopit.sim.g1_o6.config import Config
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6_support import ladder_campaign as module
    from teleopit.sim.g1_o6_support.ladder_identity import physical_identity
    from teleopit.sim.g1_o6_support.ladder_reset import select_reset
    from teleopit.sim.ladder_config import geometry_matrix

    # Shorten acquisition only inside this software test. Campaign durations,
    # controller, assistance schedule and strict thresholds remain unchanged.
    monkeypatch.setattr(
        module,
        "Config",
        lambda **kw: replace(Config(**kw), acquire_seconds=0.02, remove_seconds=0.02),
    )
    condition = module.conditions()[0]
    reset = select_reset(geometry_matrix()["G0"])
    payload = module.trial_payload(condition, reset)
    payload["reference_physics"] = physical_identity(Environment().model)
    result = module.run_condition(payload, tmp_path / "trial", 0.02, 0.001)
    output = tmp_path / "trial"
    assert not result["progression_justified"]
    assert (output / "hold_entry.json").is_file()
    with gzip.open(output / "applied_forces.jsonl.gz", "rt") as stream:
        applied = [json.loads(line) for line in stream]
    assert len(applied) == result["physics_audit"]["scored_steps"] > 0
    assert all(not any(r["qfrc_applied"]) for r in applied)
    assert all(not any(v for body in r["xfrc_applied"] for v in body) for r in applied)
    assert json.loads((output / "observed_friction.json").read_text())
    assert result["drift_assessment"]["status"] != "PASS"
