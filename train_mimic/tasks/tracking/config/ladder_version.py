"""Identity written into every ladder training log.

Bump ``LADDER_CODE_VERSION`` when the ladder task, reward, or contacts change
in a way that makes an older run incomparable.
"""

from __future__ import annotations

import os

LADDER_CODE_VERSION = "weighted-body-targets"
LADDER_CODE_VERSION_NOTES = (
    "Episodes last 8 seconds. The reset writes the climbing keyframe, then "
    "moves one limb inside a 25 cm ball. The other three limbs and the head "
    "stay on the measured points. Each point has a weight in [0, 1]: limbs "
    "start firm, one limb is softened half the time, and the head is usually "
    "a soft tether. A new target and weight vector is sampled mid-episode "
    "when the firm points stay within 3.5 cm for 12 steps or when a 1.5 to "
    "2 second subgoal timer expires. That update does not rewrite the pose. "
    "Welds stay inactive. The grip action is still two dimensions and does "
    "not latch. Each second pays 1.25 minus the capped weighted distance, "
    "plus twice the weighted drop in per-point distance. A reset, a new "
    "subgoal, and a weight that was zero on the previous step pay no drop. "
    "Ground contact pays nothing from either term and ends the episode. "
    "Action rate, self-collision, joint limits, and ankle acceleration stay "
    "on. The deadline bootstraps; touching the ground does not. Latch "
    "attachment code remains in the climb command and is not wired."
)


def record_ladder_code_version(log_dir: str) -> str:
    """Write ``code_version.txt`` into a new run directory and return the path."""

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "code_version.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{LADDER_CODE_VERSION}\n{LADDER_CODE_VERSION_NOTES}\n")
    return path
