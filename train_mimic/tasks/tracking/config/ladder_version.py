"""Identity written into every ladder training log.

Bump ``LADDER_CODE_VERSION`` when the ladder task, reward, or contacts change
in a way that makes an older run incomparable.
"""

from __future__ import annotations

import os

LADDER_CODE_VERSION = "tread-1"
LADDER_CODE_VERSION_NOTES = (
    "Ascent pays min(torso rungs, lower supported foot rung + lead0) only "
    "while both feet are on the tread and a hand is attached. lead0 is the "
    "torso rung coordinate at reset minus the initial foot rung. Foot climb "
    "aims at the tread top. A foot under the rung is not support."
)


def record_ladder_code_version(log_dir: str) -> str:
    """Write ``code_version.txt`` into a new run directory and return the path."""

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "code_version.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{LADDER_CODE_VERSION}\n{LADDER_CODE_VERSION_NOTES}\n")
    return path
