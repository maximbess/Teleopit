from dataclasses import replace

import pytest


def test_protocol_timeout_is_terminal_and_cannot_be_overwritten():
    from teleopit.sim.g1_o6_support.protocol import Protocol, Step

    protocol = Protocol([Step("OPEN", 3, 4, "opened")])
    protocol.update(0, {"opened": False})
    protocol.update(4, {"opened": False})
    assert protocol.phase == "ABORT"
    assert protocol.failure == "OPEN:timeout:opened"
    with pytest.raises(RuntimeError, match="finished"):
        protocol.update(5, {"opened": True})
    assert protocol.failure == "OPEN:timeout:opened"


def test_opening_waits_for_persistent_actual_motion():
    from teleopit.sim.g1_o6_support.protocol import Protocol, Step

    protocol = Protocol([Step("OPEN", 3, 10, "opened")])
    protocol.update(0, {"opened": False})
    protocol.update(3, {"opened": True})
    assert protocol.phase == "OPEN"
    protocol.update(3.2, {"opened": False})
    protocol.update(4, {"opened": True})
    protocol.update(4.5, {"opened": True})
    assert protocol.phase == "COMPLETE"


def test_hold_requires_whole_interval_and_release_needs_unload_first():
    from teleopit.sim.g1_o6_support.protocol import Protocol, Step, transfer_steps

    steps = transfer_steps("release", "left", 1, False, 10, 10)
    phases = [s.name for s in steps]
    assert (
        phases.index("UNLOADED_HOLD") < phases.index("OPEN") < phases.index("RETRACT")
    )
    assert phases[-1] == "THREE_SUPPORT_HOLD"
    protocol = Protocol([Step("THREE_SUPPORT_HOLD", 10, 10, "detached", hold=True)])
    protocol.update(0, {"detached": True})
    protocol.update(9, {"detached": False})
    assert protocol.phase == "ABORT"


def test_reacquisition_repeats_both_sides_and_requires_reload_hold():
    from teleopit.sim.g1_o6_support.protocol import Protocol, transfer_steps

    steps = transfer_steps("reacquire", "left", 3, True, 1, 10)
    assert [s.side for s in steps if s.name == "OPEN"] == ["left", "right", "left"]
    assert len([s for s in steps if s.name == "POST_HOLD"]) == 3
    protocol = Protocol(
        [replace(s, minimum_s=0, timeout_s=1) for s in steps], persistence_s=0
    )
    t = 0
    while protocol.phase != "COMPLETE":
        protocol.update(t, {protocol.step.guard: True})
        t += 0.1
    assert protocol.failure is None
    assert len(protocol.events) == len(steps)
