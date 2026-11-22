"""Floating regrasp through the planted legs and the supporting hand's native servos."""

import numpy as np

from teleopit.sim.o6_grasp.controller import smooth

from .g6_grip_feedback import SIDES
from .regrasp_trial import Confirmation, Trial, Unmet, WristPath

SUPPORT_ARM_FRACTION = 0.23
OPEN_RATE = 0.67
CLOSE_RATE = 0.55
SUPPORT_PRELOAD = (0.20, 0.12, 0.26, 0.26, 0.26, 0.26)
SUPPORT_TRAVEL = (0.085, 0.0, 0.20, 0.20, 0.20, 0.20)
RAMP_S = 1.5
HOLD_PRELOAD = (0.16, 0.12, 0.20, 0.20, 0.20, 0.20)
HOLD_TRAVEL = (0.045, 0.0, 0.095, 0.095, 0.095, 0.095)


class StanceServos:
    """Increase native joint impedance while preserving the static torque at capture."""

    def __init__(self, model, indices, measured, command, gain_ratio):
        self.model, self.ids = model, indices
        self.angle = measured.copy()
        self.preload = command - measured
        self.kp = model.actuator_gainprm[indices, 0].copy()
        self.kd = -model.actuator_biasprm[indices, 2].copy()
        self.compliance = 1 / gain_ratio

    def target(self, elapsed):
        fraction = smooth(elapsed / RAMP_S)
        compliance = 1.0 + fraction * (self.compliance - 1.0)
        kp = self.kp / compliance
        self.model.actuator_gainprm[self.ids, 0] = kp
        self.model.actuator_biasprm[self.ids, 1] = -kp
        self.model.actuator_biasprm[self.ids, 2] = -self.kd / np.sqrt(compliance)
        return self.angle + self.preload * compliance


def prepare_support_grip(digit, attained):
    digit.closed = np.minimum(digit.closed, attained + SUPPORT_TRAVEL)
    digit.preload = np.array(SUPPORT_PRELOAD)


class HoldingDigits:
    def __init__(self, digit):
        self.digit = digit
        self.closed, self.preload = digit.closed.copy(), digit.preload
        self.holding = False

    def select(self, hold, attained):
        if hold != self.holding:
            self.digit.closed = (
                np.minimum(self.closed, attained + HOLD_TRAVEL) if hold else self.closed
            )
            self.digit.preload = np.array(HOLD_PRELOAD) if hold else self.preload
            self.holding = hold


class HoldingWrist:
    def __init__(self, env, policy, side):
        self.path = WristPath(env, policy, side, 0)
        self.arm = env.target[self.path.ids].copy()

    def target(self, goal=None):
        # Preserve closed-chain joint stiffness while correcting motion in rung axes.
        if goal is None:
            goal = self.path.target(0.0)
        return self.arm + SUPPORT_ARM_FRACTION * (goal - self.arm)


class FloatingBalance:
    def __init__(self, env, policy, side):
        self.env = env
        self.support = HoldingWrist(env, policy, side)
        ids = policy.indices
        names = [env.model.actuator(i).name for i in ids]
        gain = np.array([3.0 if ("ankle" in n or "waist" in n) else 2.0 for n in names])
        self.servos = StanceServos(
            env.model, ids, env.data.qpos[env.act_qpos[ids]], env.target[ids], gain
        )
        self.start_s = env.data.time
        prepare_support_grip(policy.digits[side], env.target[env.hand_act[side]])
        self.returned = HoldingDigits(policy.digits[SIDES[1 - SIDES.index(side)]])

    def stance(self):
        return self.servos.target(self.env.data.time - self.start_s)

    def settings(self):
        servo, model = self.servos, self.env.model
        return {
            "start_s": self.start_s,
            "ramp_s": RAMP_S,
            "stance_actuators": [model.actuator(i).name for i in servo.ids],
            "initial_kp": servo.kp.tolist(),
            "initial_kd": servo.kd.tolist(),
            "final_kp": model.actuator_gainprm[servo.ids, 0].tolist(),
            "final_kd": (-model.actuator_biasprm[servo.ids, 2]).tolist(),
            "force_range_Nm": model.actuator_forcerange[servo.ids].tolist(),
            "stance_target_rad": self.env.target[servo.ids].tolist(),
            "holding_wrist_ik_fraction": SUPPORT_ARM_FRACTION,
            "open_rate_per_s": OPEN_RATE,
            "close_rate_per_s": CLOSE_RATE,
            "support_preload_Nm": SUPPORT_PRELOAD,
            "support_travel_rad": SUPPORT_TRAVEL,
            "holding_preload_Nm": HOLD_PRELOAD,
            "holding_travel_rad": HOLD_TRAVEL,
        }


class FloatingTrial(Trial):
    def __init__(self, env, policy, side, offset_mm):
        super().__init__(env, policy, side, "free", None, offset_mm)
        self.balance = None
        self.holding_wrist = None

    def set_stance(self):
        if self.balance is None:
            return super().set_stance()
        self.policy.stance = self.balance.stance()

    def actions(self, arm, request):
        arms, grips = super().actions(arm, request)
        if self.balance is not None:
            arms[1 - self.i] = self.balance.support.target()
            grips[1 - self.i] = 1.0
            self.balance.returned.select(
                request is None, self.env.target[self.env.hand_act[self.side]]
            )
            if request is None:
                grips[self.i] = 1.0
            else:
                grips[self.i] = request * (OPEN_RATE if request < 0 else CLOSE_RATE)
            if self.path is not None:
                arms[self.i] = self.selected_arm(arm, holding=request is None)
        return arms, grips

    def selected_arm(self, arm, *, holding):
        if not holding:
            self.holding_wrist = None
            return arm
        if self.holding_wrist is None:
            self.holding_wrist = HoldingWrist(self.env, self.policy, self.side)
        return self.holding_wrist.target(arm)

    def settle(self):
        self.acquire("support_preparation")
        self.prepare_balance()

    def prepare_balance(self, preparation_s=RAMP_S):
        self.balance = FloatingBalance(self.env, self.policy, SIDES[1 - self.i])
        timer = Confirmation(2.0, self.dt)
        start = self.time
        while self.time - start < preparation_s + 6.5:
            self.tick("prepare_support")
            after_ramp = self.last("time") >= start + preparation_s - 1e-8
            if timer.update(self.support_ready() & after_ramp):
                self.event("settled")
                self.begin_measurement()
                return
        raise Unmet(
            "two continuous seconds of four-support holding after grip preparation"
        )
