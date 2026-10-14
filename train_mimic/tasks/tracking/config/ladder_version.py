"""Identity written into every ladder training log.

Bump ``LADDER_CODE_VERSION`` when the ladder task, reward, or contacts change
in a way that makes an older run incomparable.
"""

from __future__ import annotations

import os

LADDER_CODE_VERSION = "cap-12"
LADDER_CODE_VERSION_NOTES = (
    "Climb pays 0.5 per meter of approach until the limb is on the next "
    "rung. A foot's approach is the rung-to-rung gap of the sole point "
    "furthest behind the standing height, 43 mm above the next rung center. "
    "Each limb's approach stops at 0. Ascent pays the change in "
    "clamp(torso - reset height, 0, 1). A foot on a tread that carries 20% "
    "of the robot weight banks that height, and the bank is kept. On the "
    "last step, half of the net unbanked drop in ascent and in each limb's "
    "approach is given back. A hand pays 0.25 per second while welded to "
    "the next rung. A foot on the next tread pays 0.10 per second for the "
    "contact, plus 1 per second times its share of body weight, clamped "
    "to 0.35, so the foot rate rises from 0.10 to 0.45 per second. The "
    "share is the upward force divided by mass times "
    "the simulator gravity. Each limb caps at +3. The episode ends on the 20 s timeout "
    "or when the robot touches the ground. Flight charges 1 per second "
    "only when both feet are off. Staying off the ground pays 1 per second "
    "until the episode has lasted 2 seconds, and the ground-contact step "
    "pays nothing. Gravity is one third of 9.81. The hand collision capsule "
    "and its marker share radius 0.085. The start pose sets those spheres on "
    "the near upper corner of the hand rung, clear of the ladder, and a hand "
    "attaches within 0.14 m of a rung centerline. action_rate weight is -0.002. "
    "A hold counts while all four supports stay on the next rung. It completes "
    "when that trailing second averages at most 0.20 m/s of torso speed and "
    "0.40 rad/s of torso and pelvis angular speed. Leaving a support claws "
    "the 0.5 hold potential back. Completion pays each limb the unpaid part "
    "of its +3 contact cap and then clears the cap."
)


def record_ladder_code_version(log_dir: str) -> str:
    """Write ``code_version.txt`` into a new run directory and return the path."""

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "code_version.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{LADDER_CODE_VERSION}\n{LADDER_CODE_VERSION_NOTES}\n")
    return path
