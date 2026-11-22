"""Opt-in, CPU G1/O6 arm and signed digit actions. One call is one control tick."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from teleopit.sim.g1_o6.holding import finger_targets, opening_targets

from .g6_grip_feedback import SIDES
from .g6_grip_preview import PreviewReference, alternatives, choose_grip
from .g6_wrist_grip import ThumbLandmark
from .pose_arm import rung_pose

FIXTURE = "initial_pelvis_fixture"


def fixed_model(xml, pose):
    tree = ET.fromstring(xml)
    equality = tree.find("equality")
    if equality is None:
        equality = ET.SubElement(tree, "equality")
    ET.SubElement(
        equality,
        "weld",
        name=FIXTURE,
        body1="world",
        body2="pelvis",
        relpose=" ".join(map(str, pose)),
        solref=".003 1",
        solimp=".9999 .99999 .001",
    )
    source = ET.tostring(tree, encoding="unicode")
    return mujoco.MjModel.from_xml_string(source), source


class DigitCommand:
    def __init__(self, closed, kp, preload):
        self.closed, self.kp, self.preload = np.asarray(closed), kp, preload
        self.follow(np.zeros_like(self.closed), 0.0)

    def follow(self, target, amount):
        self.amount = float(amount)
        self.target = np.array(target, copy=True)
        self.request = 0.0
        self.open_start = None

    def advance(self, request, measured, dt):
        if not np.isfinite(request):
            raise ValueError("grip request must be finite")
        self.request = float(np.clip(request, -1.0, 1.0))
        previous = self.amount
        self.amount = float(np.clip(previous + self.request * dt, 0.0, 1.0))
        if self.request < 0:
            if self.open_start is None:
                self.open_start = measured.copy(), max(previous, 1e-12)
            start, amount = self.open_start
            self.target = opening_targets(start, 1.0 - self.amount / amount)
        if self.request > 0:
            self.open_start = None
            self.target = finger_targets(
                self.amount, self.closed, measured, self.kp, self.preload
            )
        return self.target


def recapture_wrist(env, side, wrist):
    """Return feedback to the attained pose and the current command without a step."""
    ids = wrist.indices[side]
    static = wrist.base.desired(env, 1.0)
    env.body_target[ids] += env.target[ids] - static[ids]
    wrist.offset[side].fill(0.0)
    wrist.anchor[side] = wrist.reference[side] = rung_pose(env, env.data, side)
    wrist.landmarks.pop(side, None)
    if side in wrist.sides:
        from .g6_arm import feedback_contacts

        quality, _ = feedback_contacts(env)
        if quality[side]["valid"]:
            wrist.landmarks[side] = ThumbLandmark.capture(
                env, env.data, side, quality[side]
            )
    wrist.external_arms.remove(side)


class PhysicalHands(PreviewReference):
    """Install at reset on a G6 right-wrist environment; step() also returns its observations.

    Left/right arm entries are seven joint position targets in source actuator order.
    A None arm or grip entry delegates that channel to live reference feedback.
    The twelve digit servos have no joint-action entry. Environment.commands owns
    joint-range clipping, slew, queueing and the single write to data.ctrl.
    """

    @classmethod
    def install(cls, env):
        original = env.hold_controller
        if env.data.time != 0 or not isinstance(original, PreviewReference):
            raise ValueError("install physical actions at a fresh G6 reference reset")
        env.hold_controller = original.parent
        policy = cls(env)
        env.hold_controller = policy
        return policy

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.arms = dict.fromkeys(SIDES)
        self.requests = dict.fromkeys(SIDES)
        self.stance = None
        self.digits = {
            s: DigitCommand(
                env.pose.closed_pose,
                env.model.actuator_gainprm[env.hand_act[s], 0],
                env.pose.finger_preload_Nm,
            )
            for s in SIDES
        }

    def reset(self):
        self.env.reset()
        self.env.hold_controller = self.env.hold_controller.parent
        self.__init__(self.env)
        self.env.hold_controller = self
        return self

    @staticmethod
    def parse_action(arms, grip):
        arms, grip = (
            dict(zip(SIDES, arms, strict=True)),
            dict(zip(SIDES, grip, strict=True)),
        )
        for side, arm in arms.items():
            if arm is not None and (
                np.shape(arm) != (7,) or not np.isfinite(arm).all()
            ):
                raise ValueError("an arm action needs seven finite joint targets")
            if grip[side] is not None and not np.isfinite(grip[side]):
                raise ValueError("grip request must be finite")
        return arms, grip

    def action(self, arms=(None, None), grip=(None, None)):
        arms, grip = self.parse_action(arms, grip)
        for side, arm in arms.items():
            if arm is not None:
                self.parent.external_arms.add(side)
            elif side in self.parent.external_arms:
                recapture_wrist(self.env, side, self.parent)
            self.handoff_digits(side, grip[side])
        self.arms, self.requests = arms, grip

    def handoff_digits(self, side, request):
        was_automatic = self.requests[side] is None
        if was_automatic == (request is None):
            return
        attained = self.env.target[self.env.hand_act[side]]
        if was_automatic:
            digit = self.digits[side]
            digit.follow(attained, digit.amount)
        elif self.cap is not None:
            i = SIDES.index(side)
            self.cap[i] = attained
            self.grip.ceiling.reshape(2, 6)[i] = attained
            self.height_at_capture[i] = (
                self.env.data.site(self.env.hands[side]["palm"]).xpos[2]
                - self.env.data.geom(self.env.support_targets[side + "_hand"]).xpos[2]
            )

    def step(
        self,
        arms=(None, None),
        grip=(None, None),
        *,
        physics_observer=None,
        advance_phase=True,
    ):
        self.action(arms, grip)
        row = self.env.step(physics_observer, advance_phase=advance_phase)
        row["physical_hands"] = self.observe(row)
        return row

    def observe(self, row):
        from .pose_contacts import external_hand_contacts, grasp_quality

        env, result = self.env, {}
        for side in SIDES:
            site = env.data.site(env.hands[side]["mount"])
            rung = env.data.geom(env.support_targets[side + "_hand"])
            twist = np.zeros(6)
            mujoco.mj_objectVelocity(
                env.model, env.data, mujoco.mjtObj.mjOBJ_SITE, site.id, twist, 0
            )
            position, rotation = rung_pose(env, env.data, side)
            contacts = external_hand_contacts(env, row["contacts"], side)
            result[side] = {
                "position_rung_m": position.tolist(),
                "rotation_rung": rotation.tolist(),
                "twist_rung_rad_s_m_s": (twist.reshape(2, 3) @ rung.xmat.reshape(3, 3))
                .ravel()
                .tolist(),
                "measured_rad": env.data.qpos[
                    env.act_qpos[env.hand_act[side]]
                ].tolist(),
                "previous_request": self.digits[side].request,
                "contacts": contacts,
                "quality": grasp_quality(contacts, rung.xpos[2]),
            }
        return result

    def grasp_command(self, env, amount):
        command = super().grasp_command(env, amount)
        for side, digit in self.digits.items():
            ids, request = env.hand_act[side], self.requests[side]
            if request is None:
                digit.follow(command[ids], amount)
            else:
                command[ids] = digit.advance(
                    request, env.data.qpos[env.act_qpos[ids]], env.config.control_dt
                )
            if self.arms[side] is not None:
                command[self.parent.indices[side]] = self.arms[side]
        return command

    def select_offset(self, env, command, baseline, increment, bounds, frozen):
        if self.stance is None:
            return super().select_offset(
                env, command, baseline, increment, bounds, frozen
            )
        # Both branches receive the exact recorded normal-stance servo trajectory.
        # Still evaluate the normal grip alternatives on the *actual* commanded
        # arm and open digits. An alternative may only act on delegated digits.
        command[self.indices] = self.stance
        self.last = self.preview.evaluate(env, command)
        if self.cap is not None and not self.parent.inhibition:
            self.select_delegated_grip(env, command)
        return self.stance - baseline

    def select_delegated_grip(self, env, command):
        delegated = np.repeat([self.requests[s] is None for s in SIDES], 6)
        indices = self.fingers[delegated]

        def targets(cap):
            return np.minimum(self.uncapped, cap.ravel())[delegated]

        def evaluate(cap):
            trial = command.copy()
            trial[indices] = targets(cap)
            return self.preview.evaluate(env, trial)

        options = alternatives(
            self.previous_cap,
            self.nominal_cap,
            self.grip.ceiling.reshape(2, 6),
            self.uncapped.reshape(2, 6),
            env.config.control_dt,
        )
        self.grip_choice, cap, self.last, self.extra = choose_grip(
            self.last, options, evaluate
        )
        if cap is not None:
            self.cap = cap
            command[indices] = targets(cap)

    def desired(self, env, amount):
        command = super().desired(env, amount)
        # Non-yoked bilateral previews can propose a new cap late in the tick.
        for side in SIDES:
            if self.requests[side] is not None:
                command[env.hand_act[side]] = self.digits[side].target
        return command
