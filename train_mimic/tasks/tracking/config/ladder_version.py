"""Identity written into every ladder training log.

Bump ``LADDER_CODE_VERSION`` when the ladder task, reward, or contacts change
in a way that makes an older run incomparable.
"""

from __future__ import annotations

import os

LADDER_CODE_VERSION = "body-targets"
LADDER_CODE_VERSION_NOTES = (
    "Episodes last 5 seconds. The reset samples a pose from the pose bank, "
    "which currently holds the climbing keyframe, then moves one of five "
    "sites inside a 20 cm ball: both grip sites, both feet, and the head. "
    "The other four targets stay where those sites were after the reset. "
    "Welds stay inactive. The grip action is still two dimensions and does "
    "not latch. Each second pays 1.25 minus the capped sum of the five "
    "distances, plus twice the drop in that sum. The first step after a "
    "reset pays no drop. Ground contact pays nothing from either term and "
    "ends the episode. Action rate, self-collision, joint limits, and ankle "
    "acceleration stay on. The deadline bootstraps; touching the ground does "
    "not. Latch attachment code remains in the climb command and is not wired."
)


def record_ladder_code_version(log_dir: str) -> str:
    """Write ``code_version.txt`` into a new run directory and return the path."""

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "code_version.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{LADDER_CODE_VERSION}\n{LADDER_CODE_VERSION_NOTES}\n")
    return path
