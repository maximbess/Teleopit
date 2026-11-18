"""Choose a bounded hand command using bilateral opposition throughout its native interval."""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from teleopit.sim.g1_o6.diagnostics import region
from .g6_grip_feedback import GripEnvironment, GripFeedback, GripReference, MAX_RATE, SIDES, advance
from .g6_ground_reaction import BALANCE_LIMIT_N, FINGER_RESERVE_N, ContactPreview, Forecast
from .pose_contacts import FINGERS, LOADED_N, contact_group

THUMB_RESERVE_N = .50


@dataclass(frozen=True)
class PreviewGrip(GripFeedback):
    grip_law: str = "palm"
    pressure_gain: float = .12
    thumb_preview: str = "native"

    def __post_init__(self):
        super().__post_init__()
        if (self.grip_law, self.pressure_gain, self.thumb_preview) != ("palm", .12, "native"):
            raise ValueError("grip preview requires the bounded palm-height parent")


@dataclass(frozen=True)
class GraspForecast(Forecast):
    left_thumb_N: float
    right_thumb_N: float

    def healthy(self):
        return (self.peak_N < BALANCE_LIMIT_N and self.finger_N >= FINGER_RESERVE_N
                and min(self.left_thumb_N, self.right_thumb_N) >= THUMB_RESERVE_N)

    def grip_rank(self):
        loads = np.array([self.finger_N, self.left_thumb_N, self.right_thumb_N])
        hard_failures = int(self.peak_N >= BALANCE_LIMIT_N) + np.count_nonzero(loads < LOADED_N)
        deficit = np.maximum(1. - loads / [FINGER_RESERVE_N, THUMB_RESERVE_N, THUMB_RESERVE_N], 0.)
        return hard_failures, max(float(np.max(deficit)), self.peak_N / BALANCE_LIMIT_N - 1.), float(np.sum(deficit))


def choose_grip(primary, options, evaluate):
    best = ("nominal", None, primary)
    count = 0
    if not primary.healthy():
        for count, (name, cap) in enumerate(options, 1):
            forecast = evaluate(cap)
            if forecast.healthy():
                return name, cap, forecast, count
            if forecast.grip_rank() < best[2].grip_rank():
                best = name, cap, forecast
    return (*best, count)


def alternatives(previous, nominal, anchor, native, dt):
    for name, finger_rate in (("thumb", None), ("raise", MAX_RATE), ("lower", -MAX_RATE)):
        rates = np.zeros_like(previous)
        rates[:, 0] = MAX_RATE
        if finger_rate is not None:
            rates[:, 2:] = finger_rate
        cap = advance(previous, anchor, rates, native, dt)
        if finger_rate is None:
            cap[:, 2:] = nominal[:, 2:]
        yield name, cap


def hand_pairs(env):
    pairs = {}
    for side, name in enumerate(SIDES):
        rung = env.model.geom(env.support_targets[name + "_hand"]).id
        for geom, body in enumerate(env.model.geom_bodyid):
            digit = contact_group(env.model.body(body).name)
            if region(env.model, body) == name + "_hand" and digit in FINGERS:
                pairs[geom, rung] = side, digit == "thumb", -1.
                pairs[rung, geom] = side, digit == "thumb", 1.
    return pairs


def opposed_load(thumbs, fingers):
    if not thumbs or not fingers:
        return 0.
    normals, loads = zip(*thumbs)
    return float(sum(loads)) if np.min(np.asarray(normals) @ np.asarray(fingers).T) < -.5 else 0.


class BilateralPreview(ContactPreview):
    def __init__(self, env):
        super().__init__(env)
        self.hand_pairs = hand_pairs(env)

    def contacts(self, model):
        force, finger = super().contacts(model)
        thumbs, digits = [[], []], [[], []]
        wrench = np.zeros(6)
        for index, contact in enumerate(self.data.contact):
            key = self.hand_pairs.get(tuple(contact.geom))
            if key is None:
                continue
            mujoco.mj_contactForce(model, self.data, index, wrench)
            if wrench[0] >= LOADED_N:
                side, thumb, sign = key
                normal = sign * contact.frame[:3]
                if thumb:
                    thumbs[side].append((normal, wrench[0]))
                else:
                    digits[side].append(normal)
        return force, finger, *(opposed_load(t, f) for t, f in zip(thumbs, digits))

    @staticmethod
    def summarize(samples):
        forces, fingers, left, right = zip(*samples)
        return GraspForecast(float(max(forces)), float(min(fingers)), min(left), min(right))


class PreviewReference(GripReference):
    preview_type = BilateralPreview

    def __init__(self, env):
        super().__init__(env)
        self.previous_cap = self.nominal_cap = self.nominal_forecast = None
        self.grip_choice, self.extra = "nominal", 0
        self.tested_forecasts = []

    def grip_targets(self, env, elapsed):
        self.previous_cap = self.cap
        command = super().grip_targets(env, elapsed)
        self.nominal_cap = self.cap
        if self.cap is not None and self.previous_cap is None:
            self.previous_cap = self.grip.ceiling.reshape(2, 6)
        return command

    def select_offset(self, env, command, baseline, increment, bounds, frozen):
        offset = super().select_offset(env, command, baseline, increment, bounds, frozen)
        self.nominal_forecast = self.last
        self.grip_choice, self.extra = "nominal", 0
        self.tested_forecasts = []
        if self.cap is None or self.parent.inhibition or self.last.healthy():
            return offset
        def evaluate(cap):
            trial = command.copy()
            trial[self.indices] = baseline + offset
            trial[self.fingers] = np.minimum(self.uncapped, cap.ravel())
            forecast = self.preview.evaluate(env, trial)
            self.tested_forecasts.append(asdict(forecast))
            return forecast
        options = alternatives(self.previous_cap, self.nominal_cap, self.grip.ceiling.reshape(2, 6),
                               self.uncapped.reshape(2, 6), env.config.control_dt)
        self.grip_choice, cap, self.last, self.extra = choose_grip(self.last, options, evaluate)
        if cap is not None:
            self.cap = cap
            command[self.fingers] = np.minimum(self.uncapped, cap.ravel())
        return offset

    def state(self):
        return {**super().state(), "grip_preview": {"choice": self.grip_choice, "extra_evaluations": self.extra,
                "tested_forecasts": self.tested_forecasts,
                "nominal_cap_rad": None if self.nominal_cap is None else self.nominal_cap.ravel().tolist(),
                "nominal_forecast": None if self.nominal_forecast is None else asdict(self.nominal_forecast)}}

    def metadata(self, env):
        return {**super().metadata(env), "grip_preview": {"opposed_thumb_reserve_N": THUMB_RESERVE_N,
                "additional_hand_previews": 3, "horizon_s": env.config.control_dt}}


class PreviewEnvironment(GripEnvironment):
    reference_type = PreviewReference
