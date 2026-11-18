"""Staged bounded grip feedback on G6 with uninterrupted bilateral thumb opposition."""

from dataclasses import replace
from pathlib import Path

from .g6_grip_feedback import GripFeedback, RATES, SIDES
from .g6_knee_experiment import KneeCampaign, actors as knee_actors, experiment_spec
from .g6_knee_retention_experiment import RetentionObserver, candidates as retention_candidates, passes as retention_passes
from .g6_trial import run_candidate
from .g6_whole_body_experiment import main
from .pose_contacts import LOADED_N

PARENT_TRACE = "7d1e268c23112b0855a4c5f547ababdfd20e11e36f1c7dbce75b4c956aee057d"
LAWS = {"pressure_low": ("pressure", RATES[0]), "pressure_high": ("pressure", RATES[1]),
        "palm_height": ("palm", RATES[1])}
SUPPORT = ("left_thumb", "right_thumb", "left_fingers")


def candidates():
    parent = replace(retention_candidates()[-1], id="control", parent=None, family="control")
    return [parent, *[replace(parent, id=name, family=law, parent="control",
                              controller=GripFeedback(grip_law=law, pressure_gain=gain))
                      for name, (law, gain) in LAWS.items()]]


def specification():
    return {**experiment_spec("g6_grip_feedback_v1", candidates()), "parent_probe_steps_sha256": PARENT_TRACE,
            "qualification": "Every HOLD solver sample and control endpoint must retain both opposed thumbs and "
                             "all four loaded left fingers, strict positive knee clearance, residual <25 N "
                             "and the native G6 physical, grasp and stage-dependent convergence gates."}


class SupportTrace:
    def __init__(self):
        self.samples = 0
        self.loss_steps = dict.fromkeys(SUPPORT, 0)
        self.first_loss_s = dict.fromkeys(SUPPORT)

    def observe(self, elapsed, quality):
        self.samples += 1
        present = {s + "_thumb": quality[s]["thumb_opposition"] and quality[s]["thumb_normal_N"] >= LOADED_N for s in SIDES}
        present["left_fingers"] = quality["left"]["minimum_finger_normal_N"] >= LOADED_N
        for name, good in present.items():
            if not good:
                self.loss_steps[name] += 1
                if self.first_loss_s[name] is None:
                    self.first_loss_s[name] = elapsed


class GripObserver(RetentionObserver):
    def __init__(self, env, path):
        super().__init__(env, path)
        self.solver_grip, self.endpoint_grip = SupportTrace(), SupportTrace()

    def physics_reasons(self, env, row, hands, states):
        if env.phase == "HOLD":
            self.solver_grip.observe(row["time"] - env.phase_start, {s: hands[s]["quality"] for s in SIDES})
        return super().physics_reasons(env, row, hands, states)

    def enrich(self, env, row):
        super().enrich(env, row)
        if row["phase"] == "HOLD":
            self.endpoint_grip.observe(row["hold_time_s"], {s: row["pose_search"]["hands"][s]["quality"] for s in SIDES})

    def report(self):
        return {**super().report(), "grip_support": {"native": vars(self.solver_grip), "control": vars(self.endpoint_grip)}}


def continuous(trace, expected):
    return expected > 0 and trace.get("samples") == expected and trace.get("loss_steps") == dict.fromkeys(SUPPORT, 0)


def passes(result, stage):
    audit = result.get("physics_audit", {})
    grip = audit.get("grip_support", {})
    controls = audit.get("ground_reaction", {}).get("control", {}).get("samples", 0)
    return (continuous(grip.get("native", {}), audit.get("scored_steps", 0))
            and continuous(grip.get("control", {}), controls) and retention_passes(result, stage))


def run_feedback(payload, output, seconds, dt):
    return run_candidate(payload, output, seconds, dt, observer_factory=GripObserver)


def actors(record):
    return {**knee_actors(record), "grip_support": record["result"]["physics_audit"]["grip_support"]}


class GripCampaign(KneeCampaign):
    spec_factory = staticmethod(specification)
    trial_function = staticmethod(run_feedback)
    passes = staticmethod(passes)
    actors = staticmethod(actors)
    first_pair, second_pair = ("pressure_low", "pressure_high"), ("palm_height",)
    audit_key = "independent_digit_pressure"
    premise = "Independent digit load targets can stabilize both supported palms and the whole-body pose."
    replacement = "Use measured palm height to coordinate four fingers on each hand, with a local unloading reserve."
    module = "teleopit.sim.g1_o6_support.g6_grip_feedback_experiment"


if __name__ == "__main__":
    main(GripCampaign, Path("outputs/g1_o6/g6_grip_feedback"))
