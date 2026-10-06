"""Identity written into every ladder training log.

Bump ``LADDER_CODE_VERSION`` when the ladder task, reward, or contacts change
in a way that makes an older run incomparable.
"""

from __future__ import annotations

import os

LADDER_CODE_VERSION = "ground-1"
LADDER_CODE_VERSION_NOTES = (
    "Climb pays 0.5 per meter of approach until the limb is on the next "
    "rung, plus 0.25 per second while a hand is attached or a foot is on "
    "the next tread and carries 20% of the robot weight, capped at +0.25 "
    "per limb. Ascent pays the change in min(torso - reset height, 1), and "
    "a drop pays that difference back. The episode ends on the 20 s timeout "
    "or when the robot touches the ground. Flight charges only when both "
    "feet are off. action_rate weight is -0.002."
)


def record_ladder_code_version(log_dir: str) -> str:
    """Write ``code_version.txt`` into a new run directory and return the path."""

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "code_version.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{LADDER_CODE_VERSION}\n{LADDER_CODE_VERSION_NOTES}\n")
    return path
