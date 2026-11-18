"""One parser and environment factory for explicitly identified G6 controllers."""

from teleopit.sim.g1_o6.env import Environment

from .g6_acquisition import GraspAcquisition, GraspEnvironment
from .g6_arm import ArmEnvironment, ArmFeedback
from .g6_whole_body import WholeBody, WholeBodyEnvironment
from .g6_ground_reaction import GroundReaction, GroundReactionEnvironment
from .g6_knee import KneeClearance, KneeEnvironment
from .g6_knee_retention import RetainedEnvironment, RetainedKnee
from .g6_grip_feedback import GripEnvironment, GripFeedback

Controller = str | ArmFeedback | GraspAcquisition

# More specific profiles precede their parent dataclasses in both dispatches.
PROFILES = (
    ("grip_law", GripFeedback, GripEnvironment),
    ("grip_retention", RetainedKnee, RetainedEnvironment),
    ("coordination", KneeClearance, KneeEnvironment),
    ("governor", GroundReaction, GroundReactionEnvironment),
    ("balance", WholeBody, WholeBodyEnvironment),
    ("order", GraspAcquisition, GraspEnvironment),
)


def valid_controller(value):
    return value == "reference" or isinstance(value, (ArmFeedback, GraspAcquisition))


def parse_controller(value):
    if isinstance(value, dict):
        factory = next((profile for key, profile, _ in PROFILES if key in value), ArmFeedback)
        return factory(**value)
    if not valid_controller(value):
        raise ValueError("unknown G6 controller")
    return value


def environment_factory(controller):
    for _, profile, factory in PROFILES:
        if isinstance(controller, profile):
            return factory, {"grasp_acquisition": controller}
    if isinstance(controller, ArmFeedback):
        return ArmEnvironment, {"arm_feedback": controller}
    if controller != "reference":
        raise ValueError("unknown G6 controller")
    return Environment, {}


def controller_from_spec(spec):
    settings = [spec[name] for name in ("arm_feedback", "grasp_acquisition") if name in spec]
    if len(settings) > 1:
        raise ValueError("ambiguous environment controller")
    return parse_controller(settings[0] if settings else "reference")
