"""Start a floating redistribution trial at a measured four-support state under gravity."""

import json
from dataclasses import replace

import mujoco
import numpy as np

from .ladder_identity import array_digest
from .regrasp_balance import FloatingTrial, HoldingWrist
from .regrasp_measure import SUPPORTS

REFERENCE_HOLD_S = 12.04
OPEN_PRELOAD_FRACTION = 0.5
GRIP_PRELOAD_RATE_RAD_S = 0.045
GRIP_PREPARATION_S = 5.0


def require_floating_root(model):
    pelvis = model.body("pelvis")
    root = pelvis.jntadr[0]
    if not (
        pelvis.parentid[0] == 0
        and pelvis.jntnum[0] == 1
        and model.jnt_type[root] == mujoco.mjtJoint.mjJNT_FREE
    ):
        raise ValueError("The pelvis must have a six-DOF free root.")
    if model.nmocap or np.any(model.body_gravcomp):
        raise ValueError(
            "An unassisted trial requires zero mocap bodies and gravity compensation."
        )
    if (
        model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY
        or model.opt.gravity[2] >= 0
    ):
        raise ValueError("The free-body experiment requires downward gravity.")


def finger_couplings(model, hands):
    expected = sorted((p, a) for hand in hands.values() for p, a, _ in hand["passive"])
    pairs = []
    for kind, a, b in zip(model.eq_type, model.eq_obj1id, model.eq_obj2id, strict=True):
        if kind != mujoco.mjtEq.mjEQ_JOINT or min(a, b) < 0:
            raise ValueError(
                "External fixture or single-joint lock in the floating model."
            )
        pairs.append((model.joint(a).name, model.joint(b).name))
    if sorted(pairs) != expected:
        raise ValueError("Only the declared internal O6 finger couplings are allowed.")
    return pairs


def free_body_setup(model, hands):
    require_floating_root(model)
    return {
        "free_root_dofs": 6,
        "internal_finger_couplings": [list(p) for p in finger_couplings(model, hands)],
        "mocap_bodies": model.nmocap,
        "body_gravity_compensation": 0.0,
        "gravity_m_s2": model.opt.gravity.tolist(),
        "tendon_count": model.ntendon,
    }


def launch_state(frame, joint_names):
    force = [
        frame["assistance_gain"],
        *frame["assistance_world"],
        *frame["disturbance_world"],
    ]
    if (
        frame["phase"] != "HOLD"
        or np.any(force)
        or not all(frame["supports"][n]["loaded"] for n in SUPPORTS)
    ):
        raise ValueError("The start frame must be an unassisted four-support hold.")
    joints = frame["diagnostic_v1"]["joints"]
    qpos = np.array([*frame["base_pose"], *frame["joint_positions"]])
    qvel = np.array(
        [*frame["base_velocity"], *(joints[n]["velocity_rad_s"] for n in joint_names)]
    )
    return qpos, qvel, np.asarray(frame["applied_position_targets_rad"])


def restore_hold(env, policy, source):
    setup = free_body_setup(env.model, env.hands)
    with (source / "steps.jsonl").open() as stream:
        frame = next(
            (
                row
                for row in map(json.loads, stream)
                if abs(row["time"] - REFERENCE_HOLD_S) < 1e-8
            ),
            None,
        )
    if frame is None:
        raise ValueError(f"Source has no verified hold frame at {REFERENCE_HOLD_S} s.")
    names = [env.model.joint(i).name for i in range(1, env.model.njnt)]
    qpos, qvel, ctrl = launch_state(frame, names)
    if env.config.delay_steps or env.model.na:
        raise ValueError(
            "This reference requires stateless actuators and zero target delay."
        )
    env.config = replace(
        env.config, assistance_force_N=0.0, assistance_moment_Nm=0.0, disturbance_N=0.0
    )
    data = env.data
    mujoco.mj_resetData(env.model, data)
    data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, ctrl
    env.target = ctrl.copy()
    env.assistance_gain = 0.0
    env.transition("HOLD")
    mujoco.mj_forward(env.model, data)
    for side, digit in policy.digits.items():
        digit.follow(ctrl[env.hand_act[side]], 1.0)
    return {
        **setup,
        "source_time_s": frame["time"],
        "start_time_s": data.time,
        "qpos": qpos.tolist(),
        "qvel": qvel.tolist(),
        "ctrl": ctrl.tolist(),
        "state_sha256": array_digest([("qpos", qpos), ("qvel", qvel), ("ctrl", ctrl)]),
        "reference_Fz_N": [frame["supports"][s]["force_world"][2] for s in SUPPORTS],
    }


class UnassistedTrial(FloatingTrial):
    def __init__(self, env, policy, side, offset_mm, source):
        super().__init__(env, policy, side, offset_mm)
        self.launch = restore_hold(env, policy, source)
        self.launch.update(
            opening_arm_preload_fraction=OPEN_PRELOAD_FRACTION,
            grip_preload_rate_rad_s=GRIP_PRELOAD_RATE_RAD_S,
            grip_preparation_s=GRIP_PREPARATION_S,
        )

    def actions(self, arm, request):
        arms, grips = super().actions(arm, request)
        if request is not None and request < 0:
            arms[self.i] = arm - (1 - OPEN_PRELOAD_FRACTION) * self.path.preload
        return arms, grips

    def settle(self):
        self.event("support_preparation")
        self.holding_wrist = HoldingWrist(self.env, self.policy, self.side)
        self.path = self.holding_wrist.path
        self.policy.grip_slew_rad_s = GRIP_PRELOAD_RATE_RAD_S
        self.prepare_balance(GRIP_PREPARATION_S)
        self.policy.grip_slew_rad_s = None
