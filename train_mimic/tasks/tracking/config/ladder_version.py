"""Identity written into every ladder training log.

Bump ``LADDER_CODE_VERSION`` when the ladder task, reward, or contacts change
in a way that makes an older run incomparable.
"""

from __future__ import annotations

import os

LADDER_CODE_VERSION = "contact-1"
LADDER_CODE_VERSION_NOTES = (
    "Climb pays 0.25 once when a hand welds to the next rung with both feet "
    "on a tread, or a foot plants the next tread with a hand attached. Air "
    "distance pays nothing. Ascent still pays min(torso, lower foot + lead0) "
    "only while both feet are on a tread and a hand is attached. Flight "
    "charges only when both feet are off. action_rate weight is -0.002."
)


def record_ladder_code_version(log_dir: str) -> str:
    """Write ``code_version.txt`` into a new run directory and return the path."""

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "code_version.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{LADDER_CODE_VERSION}\n{LADDER_CODE_VERSION_NOTES}\n")
    return path
