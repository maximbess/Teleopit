"""Bounded native-contact preview for the G6 ground-reaction position controller."""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from teleopit.sim.g1_o6.diagnostics import region
from teleopit.sim.g1_o6.env import bounded
from teleopit.sim.g1_o6.holding import COMMAND_RATE_RAD_S

from .g6_whole_body import WholeBody, WholeBodyEnvironment, WholeBodyReference, bounded_offset
from .pose_contacts import FINGERS, LOADED_N, contact_group

GUARDS_N = (16., 21.)
FINGER_RESERVE_N = .35
TRIMS_N = (.5, -.5, 2., -2., 8., -8.)
BALANCE_LIMIT_N = 25.


@dataclass(frozen=True)
class GroundReaction(WholeBody):
    balance: str = "wrench"
    gain: float = 1.6
    governor: str = "collective"
    guard_N: float = GUARDS_N[0]

    def __post_init__(self):
        super().__post_init__()
        if (self.balance, self.gain) != ("wrench", 1.6):
            raise ValueError("refinement requires the measured high-gain ground-reaction parent")
        if self.governor not in ("collective", "individual") or self.guard_N not in GUARDS_N:
            raise ValueError("unknown bounded ground-reaction refinement")


@dataclass(frozen=True)
class Forecast:
    peak_N: float
    finger_N: float

    def rank(self, guard):
        violation = max(0., self.peak_N / BALANCE_LIMIT_N - 1., 1. - self.finger_N / LOADED_N)
        unsafe = self.peak_N >= BALANCE_LIMIT_N or self.finger_N < LOADED_N
        return unsafe, violation, max(self.peak_N / guard, FINGER_RESERVE_N / max(.001, self.finger_N))


def choose_preview(options, evaluate, guard):
    best = None
    for count, option in enumerate(options, 1):
        forecast = evaluate(option[1])
        if forecast.peak_N < guard and forecast.finger_N >= FINGER_RESERVE_N:
            return option, forecast, count
        score = forecast.rank(guard)
        if best is None or score < best[0]:
            best = (score, option, forecast)
    return best[1], best[2], count


class ContactPreview:
    def __init__(self, env):
        if env.config.disturbance_N:
            raise ValueError("ground-reaction preview requires an undisturbed HOLD")
        model = env.model
        self.data = mujoco.MjData(model)
        self.weight = env.mass * model.opt.gravity
        self.limits = model.jnt_range[model.actuator_trnid[:, 0]]
        self.environment = np.array([int(region(model, b) == "environment") for b in model.geom_bodyid])
        self.digit = np.full((model.ngeom, model.ngeom), -1, dtype=int)
        target = model.geom(env.support_targets["left_hand"]).id
        for geom, body in enumerate(model.geom_bodyid):
            name = contact_group(model.body(body).name)
            if region(model, body) == "left_hand" and name in FINGERS[1:]:
                self.digit[geom, target] = self.digit[target, geom] = FINGERS[1:].index(name)

    def contacts(self, model):
        net, fingers, wrench = self.weight.copy(), np.zeros(4), np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            a, b = contact.geom
            sign = self.environment[a] - self.environment[b]
            if sign:
                mujoco.mj_contactForce(model, self.data, i, wrench)
                net += sign * (wrench[:3] @ contact.frame.reshape(3, 3))
                digit = self.digit[a, b]
                if digit >= 0 and wrench[0] >= LOADED_N:
                    fingers[digit] += wrench[0]
        return np.linalg.norm(net), min(fingers)

    def evaluate(self, env, command):
        data, model = self.data, env.model
        mujoco.mj_copyData(data, model, env.data)
        # The copy simulates the same unassisted interval and native actuator
        # clipping, slew and delay as Environment.commands. Live data is read-only.
        limited = np.clip(command, self.limits[:, 0], self.limits[:, 1])
        rate = COMMAND_RATE_RAD_S * env.config.control_dt
        data.ctrl[:] = env.queue[0] if env.queue else env.target + np.clip(limited - env.target, -rate, rate)
        data.qfrc_applied[:] = 0.
        data.xfrc_applied[:] = 0.
        samples = []
        for _ in range(round(env.config.control_dt / env.config.physics_dt)):
            mujoco.mj_step(model, data)
            samples.append(self.contacts(model))
        mujoco.mj_forward(model, data)
        samples.append(self.contacts(model))
        return self.summarize(samples)

    @staticmethod
    def summarize(samples):
        balances, fingers = zip(*samples)
        return Forecast(float(max(balances)), float(min(fingers)))


class GroundReactionReference(WholeBodyReference):
    preview_type = ContactPreview

    def __init__(self, env):
        super().__init__(env)
        self.preview = self.preview_type(env)
        self.last, self.choice, self.count = None, None, 0

    def leg_targets(self, env, levers, force, moment):
        targets = super().leg_targets(env, levers, force, moment)
        self.wrench = levers, force, moment, np.concatenate(targets)
        return targets

    def options(self, env, step, bounds, frozen):
        dt = env.config.control_dt
        for name, fraction in (("nominal", 1.), ("half", .5), ("hold", 0.)):
            yield name, self.offset + fraction * step
        levers, force, moment, original = self.wrench
        sites = ("left", "right", "both") if self.profile.governor == "individual" else ("both",)
        for z in TRIMS_N:
            vertical = np.array([0., 0., z])
            for site in sites:
                couple = np.zeros(3) if site == "both" else np.cross(levers[site], vertical)
                targets = super().leg_targets(env, levers, bounded(force + vertical, 30.), bounded(moment + couple, 5.))
                trim = np.concatenate(targets) - original
                increment = step.copy()
                increment[:len(trim)] += trim
                yield f"{site}_{z:+g}N", bounded_offset(self.offset, increment, bounds, frozen, dt)

    def select_offset(self, env, command, baseline, increment, bounds, frozen):
        nominal = super().select_offset(env, command, baseline, increment, bounds, frozen)
        def evaluate(offset):
            trial = command.copy()
            trial[self.indices] = baseline + offset
            return self.preview.evaluate(env, trial)
        # The parent's structural/slip backoff has priority over any alternative.
        options = [("backoff", nominal)] if self.parent.inhibition else self.options(env, nominal-self.offset, bounds, frozen)
        (self.choice, offset), self.last, self.count = choose_preview(options, evaluate, self.profile.guard_N)
        return offset

    def state(self):
        return {**super().state(), "ground_reaction_preview": {"choice": self.choice, "evaluations": self.count,
                "forecast": asdict(self.last) if self.last else None}}

    def metadata(self, env):
        return {**super().metadata(env), "ground_reaction_governor": asdict(self.profile),
                "preview_horizon_s": env.config.control_dt, "finger_reserve_N": FINGER_RESERVE_N,
                "vertical_trim_choices_N": list(TRIMS_N), "preview_score": "peak native and endpoint residual; minimum left intended-tread finger force"}


class GroundReactionEnvironment(WholeBodyEnvironment):
    reference_type = GroundReactionReference
