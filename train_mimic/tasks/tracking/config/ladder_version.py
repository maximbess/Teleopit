"""Identity written into every ladder training log.

Bump ``LADDER_CODE_VERSION`` when the ladder task, reward, or contacts change
in a way that makes an older run incomparable.
"""

from __future__ import annotations

import os

LADDER_CODE_VERSION = "cap-4"
LADDER_CODE_VERSION_NOTES = (
    "Climb pays 0.5 per meter of approach until the limb is on the next "
    "rung. A foot's approach is the rung-to-rung gap of the sole point "
    "furthest behind the standing height, 43 mm above the next rung center. "
    "Each limb's approach stops at 0. Ascent pays the change in "
    "clamp(torso - reset height, 0, 1). A foot on a tread that carries 20% "
    "of the robot weight banks that height, and the bank is kept. On the "
    "last step, half of the net unbanked drop in ascent and in each limb's "
    "approach is given back. A hand pays 0.25 per second while welded to "
    "the next rung, and a loaded foot on the next tread pays 0.25 per "
    "second, capped at +3 per limb. The episode ends on the 20 s timeout "
    "or when the robot touches the ground. Flight charges 1 per second "
    "only when both feet are off. action_rate weight is -0.002."
)


def record_ladder_code_version(log_dir: str) -> str:
    """Write ``code_version.txt`` into a new run directory and return the path."""

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "code_version.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{LADDER_CODE_VERSION}\n{LADDER_CODE_VERSION_NOTES}\n")
    return path
