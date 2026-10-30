"""Identity written into every ladder training log.

Bump ``LADDER_CODE_VERSION`` when the ladder task, reward, or contacts change
in a way that makes an older run incomparable.
"""

from __future__ import annotations

import os

LADDER_CODE_VERSION = "weld-100n"
LADDER_CODE_VERSION_NOTES = (
    "The reset stands on rung 3 with the mass over the tread, knees nearly "
    "straight, and both hands latched on rung 7. Each weld stays under 100 N "
    "and the pelvis stays under the overspeed trip. "
    "The climb return is logged as four limb terms: left arm, right arm, "
    "left leg, and right leg. A rung is the stages in CLIMB_PHASE_ORDER. "
    "The default order is the lead foot, the other foot, the lead hand, "
    "the other hand, then the hold. The stage at index i pays 2**i times "
    "the first stage, on its approach, contact, detach, contact cap, and "
    "finish. Alive, ungrip, latch pressure, flight, stance, both cross "
    "terms, and action rate stay unscaled. Swapping the limb entries "
    "moves those scales. lead_hand only chooses the side. A foot phase "
    "ends after that "
    "foot has held the next rung for one second of inward contact at 20% "
    "of body weight. The plant clock scales that heading by pressure, "
    "from 0 at no load to 1 at 20%, and heavier contact stays at 1. "
    "The unscaled finish is +4 and the unscaled cap is +3. "
    "The plant clock adds the toe's horizontal heading into the ladder. "
    "Inward is the rung box's far side, orthogonal to the rung. Both that "
    "direction and the heel-toe axis drop their vertical part before the "
    "dot product. Straight in "
    "counts in full, along the rung or straight down counts as zero, and "
    "straight out runs the clock backward. A hand phase ends when that hand "
    "latches the next rung. That step fills whatever "
    "is left of that hand's unscaled +3 contact cap. Both limbs of a pair "
    "on one step pay the sum of their scaled finishes and fill both caps. "
    "A climb column is on only during its phase, "
    "and locks off once that phase ends. The climbing hand pays 0.1 per second "
    "times grip progress, so a closed weld costs 0.1 per second. Every "
    "other hand pays (progress - 1) * 0.3 per second. "
    "The hold runs only after both feet and both hands have passed their "
    "gates, and a finished hold starts the next rung at the first stage. "
    "Climb pays 0.5 per meter of approach along the plane through the two "
    "lowest rungs, up to the far top edge of the next rung. That edge is the "
    "far top corner of the rung box, read from the rung geom. Passing the "
    "edge adds nothing. Distance out of that plane costs 0.5 per meter from "
    "the attached pose, which is the zero. A foot's approach is the path "
    "around the next rung box to the standing slab, using the sole point "
    "still outside it. The slab is the top of that box at 43 mm above the "
    "center, and local +x of the rung geom is the far side. "
    "Each limb's approach stops at 0. On the "
    "last step, half of the net unbanked drop in each limb's "
    "approach is given back. Grip progress stays 0 outside the 0.14 m latch "
    "zone. A saturated command takes one second to close or open. The weld "
    "closes at progress 1 and opens at 0. A closed weld also opens when the "
    "norm of its three translational constraint forces exceeds 100 N. The "
    "rotational rows are not part of that load, and each hand trips on its "
    "own. A latched hand on the next rung "
    "pays 0.25 per second. An unlatched hand in that zone pays the same rate "
    "times progress. Both share the +3 cap. A foot on the next tread pays "
    "0.10 per second for the contact, plus 1 per second times its share of "
    "body weight times that same inward dot product. The share is "
    "clamped to 0.35. A toe aimed straight in rises from 0.10 "
    "to 0.45 per second, a toe along the rung pays the contact only, and a "
    "toe aimed straight out subtracts the pressure term. "
    "The share is the downward contact force, the negation of "
    "world z, divided by mass times "
    "the simulator gravity. Each limb caps at +3. The episode ends on the 20 s timeout "
    "or when the robot touches the ground. Staying off the ground pays 1 per second "
    "until the episode has lasted 2 seconds, and the ground-contact step "
    "pays nothing. Gravity is one third of 9.81. The hand collision capsule "
    "and its marker share radius 0.085. "
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
