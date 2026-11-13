"""Detailed windows follow acquisition transitions as well as HOLD failures."""

import json
from dataclasses import replace


def test_default_ladder_observer_triggers_every_acquisition_phase(tmp_path):
    from teleopit.sim.g1_o6.config import Config
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6_support.ladder_observer import LadderObserver

    env = Environment(replace(Config(), acquire_seconds=0.02, remove_seconds=0.02))
    observer = LadderObserver(env, tmp_path)
    try:
        for _ in range(4):
            env.step(observer)
    finally:
        observer.close()
    triggers = json.loads((tmp_path / "physics_window_triggers.json").read_text())
    reasons = {reason for trigger in triggers for reason in trigger["signals"]}
    expected = {
        "phase_transition:" + phase
        for phase in ("INITIALIZE", "ACQUIRE", "REMOVE_ASSISTANCE", "HOLD")
    }
    assert expected <= reasons
