"""HOLD-entry reference interventions around the unchanged physical servos."""

from teleopit.sim.g1_o6.metadata import controller_settings

MODES = ("reference", "freeze_thumb", "freeze_all", "cap_thumb")
SUPPORTED_MODES = (*MODES, "arm_feedback")


class FingerExperiment:
    def __init__(self, env, mode="reference"):
        if mode not in MODES:
            raise ValueError("unknown finger experiment")
        self.mode, self.base = mode, env.hold_controller
        self.thumb = {
            s: next(
                i
                for i in indices
                if env.model.actuator(i).name.endswith("_thumb_cmc_yaw_servo")
            )
            for s, indices in env.hand_act.items()
        }
        self.reset(env)

    def reset(self, env):
        self.latch, self.cap_active = {}, {}
        self.latch_time = None
        self.raw = env.data.ctrl.copy()

    def desired(self, env, amount):
        desired = self.base.desired(env, amount)
        if env.phase == "HOLD" and self.mode != "reference":
            self.latch_once(env)
            for side, indices in env.hand_act.items():
                self.intervene(desired, side, indices)
        self.raw = desired.copy()
        return desired

    def latch_once(self, env):
        if not self.latch:
            self.latch = {s: env.data.ctrl[a].copy() for s, a in env.hand_act.items()}
            self.latch_time = float(env.data.time)

    def intervene(self, desired, side, indices):
        if self.mode == "freeze_all":
            desired[indices] = self.latch[side]
            return
        index = self.thumb[side]
        latched = self.latch[side][list(indices).index(index)]
        if self.mode == "freeze_thumb":
            desired[index] = latched
        else:
            cap = latched + 0.03
            self.cap_active[side] = bool(desired[index] > cap)
            desired[index] = min(desired[index], cap)

    def state(self):
        return {
            "mode": self.mode,
            "latch_time_s": self.latch_time,
            "applied_targets_at_hold_entry_rad": {
                s: v.tolist() for s, v in self.latch.items()
            },
            "thumb_cap_active": self.cap_active.copy(),
        }

    def metadata(self, env):
        value = controller_settings(env.pose, env.config)
        if self.mode != "reference":
            value.update(
                schema="o6_hold_reference_v1",
                mode=self.mode,
                thumb_cap_rad=0.03,
                latch="actual applied targets independently at HOLD entry",
                reset="clear latches on every fresh reset",
                isolated_runner_support=False,
            )
        return value
