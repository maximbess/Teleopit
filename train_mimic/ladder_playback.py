"""Shared phase-prefix selection for ladder playback entry points."""

from __future__ import annotations

from typing import Any


LADDER_PLAY_PHASES = {
    "stabilize": 0,
    "first_hand": 1,
    "second_hand": 2,
    "first_foot": 3,
    "second_foot": 4,
}
DEFAULT_LADDER_PLAY_PHASE = "second_foot"


def configure_ladder_play_phase(
    env_cfg: Any,
    phase_name: str | None,
    *,
    freeze_at_boundary: bool = False,
) -> str:
    """Set the deepest enabled ordered ladder phase and return its name."""

    selected = phase_name or DEFAULT_LADDER_PLAY_PHASE
    try:
        phase = LADDER_PLAY_PHASES[selected]
    except KeyError as exc:
        choices = ", ".join(LADDER_PLAY_PHASES)
        raise ValueError(
            f"Unknown ladder play phase {selected!r}; expected one of: {choices}"
        ) from exc

    command_cfg = env_cfg.commands.get("ladder")
    if command_cfg is None:
        raise RuntimeError("Ladder play config does not contain the 'ladder' command")
    command_cfg.fixed_max_unlocked_phase = phase
    command_cfg.freeze_at_max_unlocked_phase = freeze_at_boundary
    return selected
