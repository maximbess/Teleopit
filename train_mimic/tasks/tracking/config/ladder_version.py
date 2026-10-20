"""Identity written into every ladder training log.

Bump ``LADDER_CODE_VERSION`` when the ladder task, reward, or contacts change
in a way that makes an older run incomparable.
"""

from __future__ import annotations

import os

LADDER_CODE_VERSION = "phase-cap-29"
LADDER_CODE_VERSION_NOTES = (
    "The climb return is logged as four limb terms: left arm, right arm, "
    "left leg, and right leg. A rung is five stages. lead_hand chooses "
    "which hand climbs first, then the other hand, then that same foot, "
    "then the other foot, then the hold. A hand phase ends when that hand "
    "latches the next rung and pays +4, and that step also fills whatever "
    "is left of that hand's +3 contact cap. A foot phase ends when that "
    "foot is on the next rung with 20% of body weight and pays +4, filling "
    "that foot's cap the same way. Both limbs of a pair on one step pays "
    "+8 and fills both caps. Later contact on a filled limb pays nothing "
    "until the next rung. The climbing hand pays 0.1 per second "
    "times grip progress, so a closed weld costs 0.1 per second. Every "
    "other hand pays (progress - 1) * 0.3 per second. Every other foot "
    "pays (min(share / 0.20, 1) - 1) * 0.3 per second, so a foot at 20% of "
    "body weight pays 0 and a foot with no load pays -0.3 per second. The "
    "stepping foot is exempt, and both feet pay during the hand and hold "
    "phases. "
    "The hold runs only after both feet have passed that gate, and a "
    "finished hold starts the next rung at the lead hand. "
    "Climb pays 0.5 per meter of approach along the plane through the two "
    "lowest rungs, up to the far top edge of the next rung. That edge is the "
    "far top corner of the rung box, read from the rung geom. Passing the "
    "edge adds nothing. Distance out of that plane costs 0.5 per meter from "
    "the attached pose, which is the zero. A foot's approach is the rung-to-rung gap of the sole "
    "point furthest behind the standing height, 43 mm above the next rung "
    "center. "
    "Each limb's approach stops at 0. On the "
    "last step, half of the net unbanked drop in each limb's "
    "approach is given back. Grip progress stays 0 outside the 0.14 m latch "
    "zone. A saturated command takes one second to close or open. The weld "
    "closes at progress 1 and opens at 0. A latched hand on the next rung "
    "pays 0.25 per second. An unlatched hand in that zone pays the same rate "
    "times progress. Both share the +3 cap. A foot on the next tread pays "
    "0.10 per second for the contact, plus 1 per second times its share of "
    "body weight, clamped to 0.35, so the foot rate rises from 0.10 to 0.45 "
    "per second. The share is the downward contact force, the negation of "
    "world z, divided by mass times "
    "the simulator gravity. Each limb caps at +3. The episode ends on the 20 s timeout "
    "or when the robot touches the ground. Staying off the ground pays 1 per second "
    "until the episode has lasted 2 seconds, and the ground-contact step "
    "pays nothing. Gravity is one third of 9.81. The hand collision capsule "
    "and its marker share radius 0.085. The start pose sets those spheres on "
    "the near upper corner of the hand rung, clear of the ladder. "
    "action_rate weight is -0.002. "
    "A hold counts during the hold phase while all four supports stay on the "
    "next rung. It completes "
    "when that trailing second averages at most 0.20 m/s of torso speed and "
    "0.40 rad/s of torso and pelvis angular speed. Leaving a support claws "
    "the 0.5 hold potential back. Completion clears the contact caps. A projected leg "
    "crossing costs 0.5 per second, and the episode stops at -1.2. "
    "A projected crossing of a leg and a climbing-face rail costs 2 per "
    "second, and the episode stops at -4.8. "
    "Loading a closed weld past half the release speed costs 32 per second "
    "times the squared excess, so the trip itself costs 8 per second. "
    "Two or more limbs off the climb cost 1 per second, and that sum stops at 1. "
    "The actor command is the two latch bits followed by the two progress values."
)


def record_ladder_code_version(log_dir: str) -> str:
    """Write ``code_version.txt`` into a new run directory and return the path."""

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "code_version.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{LADDER_CODE_VERSION}\n{LADDER_CODE_VERSION_NOTES}\n")
    return path
