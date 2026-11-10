"""Read-only control-rate telemetry and all-link momentum-rate residuals."""

import numpy as np

from .mechanics import (
    actuator_details,
    contact_rows,
    contact_wrench,
    joint_details,
    momentum,
    pose_change,
    relative_pose,
    shift_wrench,
    support_details,
)


class Telemetry:
    def __init__(self, model, data, base, bodies, groups, targets, hands):
        self.model = model
        self.base = base
        self.bodies = bodies
        self.groups = groups
        self.targets = targets
        self.hands = hands
        self.reset_reference = self.reference(data)
        self.hold_reference = None
        self.previous = None

    def reference(self, data):
        return data.body(self.base).xpos.copy(), data.body(self.base).xmat.reshape(
            3, 3
        ).copy()

    def start_hold(self, data):
        if self.hold_reference is None:
            self.hold_reference = self.reference(data)

    def sample(self, data, desired, limited, applied_wrenches):
        m = self.model
        rows = contact_rows(m, data)
        motion = momentum(m, data, self.bodies)
        center = np.array(motion["com_world_m"])
        gravity = np.r_[m.body_mass[self.bodies].sum() * m.opt.gravity, np.zeros(3)]
        contact = contact_wrench(rows, self.bodies, center)
        external = (
            contact
            + gravity
            + sum(
                (shift_wrench(w, p, center) for w, p in applied_wrenches), np.zeros(6)
            )
        )
        position, rotation = self.reference(data)
        result = {
            "schema": "o6_telemetry_v1",
            "time_s": float(data.time),
            **joint_details(m, data),
            "actuators": actuator_details(m, data, desired, limited),
            "contacts": rows,
            "from_reset": pose_change(position, rotation, self.reset_reference),
            "from_hold": pose_change(position, rotation, self.hold_reference),
            "momentum": motion,
            "contact_wrench_world_about_com": contact.tolist(),
            "gravity_wrench_world_about_com": gravity.tolist(),
            "external_wrench_world_about_com": external.tolist(),
            "momentum_rate": self.rate(data.time, motion, external),
            "supports": self.supports(rows, data),
            "rung_relative_poses_v1": self.poses(data),
            "penetration_m": max(
                0.0, max((-r["distance_m"] for r in rows), default=0.0)
            ),
            "warnings": [int(w.number) for w in data.warning],
        }
        return result

    def rate(self, time, motion, external):
        values = np.r_[motion["linear_kg_m_s"], motion["angular_kg_m2_s"]]
        result = {
            "linear_N": None,
            "angular_Nm": None,
            "balance_residual_N_Nm": None,
            "interval_s": None,
            "method": "unfiltered backward difference; compare trapezoidal endpoint external wrench; contact transients have O(dt) error",
        }
        if self.previous is not None and time > self.previous[0]:
            t, old, force = self.previous
            interval = float(time - t)
            rate = (values - old) / interval
            result.update(
                linear_N=rate[:3].tolist(),
                angular_Nm=rate[3:].tolist(),
                balance_residual_N_Nm=((external + force) / 2 - rate).tolist(),
                interval_s=interval,
            )
        self.previous = (float(time), values.copy(), external.copy())
        return result

    def supports(self, rows, data):
        return {
            name: support_details(
                rows, self.groups[name], target, data.geom(target).xpos
            )
            for name, target in self.targets.items()
        }

    def poses(self, data):
        result = {}
        for side, hand in self.hands.items():
            target = self.targets.get(side + "_hand")
            if target is None:
                continue
            rung = data.geom(target)
            result[side] = {
                name: relative_pose(
                    data.site(hand[name]).xpos,
                    data.site(hand[name]).xmat.reshape(3, 3),
                    rung.xpos,
                    rung.xmat.reshape(3, 3),
                )
                for name in ("mount", "palm")
            }
        return result
