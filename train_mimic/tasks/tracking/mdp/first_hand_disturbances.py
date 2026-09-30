"""Per-world gravity and finite force pulses for the fixed-ladder task."""
from __future__ import annotations

import torch
import warp as wp


class FirstHandDisturbances:
    def __init__(self, command):
        self.c = command
        self.cfg = command.cfg
        env = command._env
        self.device = env.device
        self.n = env.num_envs
        self.nominal_gravity = torch.tensor(env.sim.mj_model.opt.gravity.copy(), device=self.device, dtype=torch.float32)
        # opt.gravity is a nested Warp field; mjlab.expand_model_fields only
        # handles top-level MjModel fields. Expand before any episode is run,
        # then rebuild graphs so they capture the new buffer address.
        if env.sim.wp_model.opt.gravity.shape[0] != self.n:
            values = self.nominal_gravity.repeat(self.n, 1).cpu().numpy()
            env.sim.wp_model.opt.gravity = wp.array(values, dtype=wp.vec3, device=env.sim.wp_device)
            env.sim.model.clear_cache()
            env.sim.create_graph()
        self.gravity = wp.to_torch(env.sim.wp_model.opt.gravity)
        self.weight = float(env.sim.mj_model.body_subtreemass[command._pelvis_body_id]) * float(self.nominal_gravity.norm())
        self.clean = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        self.gravity_scale = torch.ones(self.n, device=self.device)
        self.remaining = torch.zeros(self.n, device=self.device)
        self.cooldown = torch.zeros_like(self.remaining)
        self.force = torch.zeros(self.n, 3, device=self.device)
        self.push_count = torch.zeros_like(self.remaining)
        self.peak_force = torch.zeros_like(self.remaining)
        self.challenge_started = torch.zeros_like(self.clean)
        self.challenge_done = torch.zeros_like(self.clean)
        self.challenge_active = torch.zeros_like(self.clean)

    def uniform(self, ids, limits):
        return torch.empty(len(ids), device=self.device).uniform_(*limits)

    def reset(self, ids):
        self.clean[ids] = torch.rand(len(ids), device=self.device) < self.cfg.clean_episode_probability
        self.gravity_scale[ids] = torch.where(
            self.clean[ids], 1., self.uniform(ids, self.cfg.gravity_scale_range))
        self.gravity[ids] = self.gravity_scale[ids, None] * self.nominal_gravity
        self.cooldown[ids] = self.uniform(ids, self.cfg.push_interval_s)
        for buf in (self.remaining, self.force, self.push_count, self.peak_force,
                    self.challenge_started, self.challenge_done, self.challenge_active):
            buf[ids] = 0
        self.write_force(ids)

    @property
    def recovery_ready(self):
        return self.clean | (self.challenge_done & (self.remaining <= 0))

    def write_force(self, ids):
        # Force at the torso COM in world coordinates; no velocity teleport.
        self.c._env.sim.data.xfrc_applied[ids, self.c._torso_body_id, :3] = self.force[ids]
        self.c._env.sim.data.xfrc_applied[ids, self.c._torso_body_id, 3:] = 0.

    def tick(self, dt):
        c = self.c
        live = ~c.motion_failed & ~c.finished & ~c._fresh_reset
        active = self.remaining > 0
        self.remaining[:] = (self.remaining-dt).clamp_min(0)
        expired = active & (self.remaining <= 0)
        self.challenge_done |= expired & self.challenge_active
        self.challenge_active[expired] = False
        self.force[expired] = 0
        self.cooldown -= dt
        # Stage values are fixed by FirstHandStage: prepare=1, release=2,
        # transfer=3, hold=4. Only one test pulse is scheduled during HOLD,
        # leaving a quiet recovery window instead of restarting it forever.
        hold = c.motion_stage == 4
        test = hold & ~self.challenge_started & (c.stage_elapsed >= self.cfg.hold_push_delay_s)
        regular = (c.motion_stage >= 1) & (c.motion_stage <= 3) & (self.cooldown <= 0)
        eligible = live & ~self.clean & (self.remaining <= 0) & (test | regular)
        ids = eligible.nonzero().flatten()
        angle = self.uniform(ids, (0., 2*torch.pi))
        strength = self.uniform(ids, self.cfg.push_weight_fraction) * self.weight
        self.force[ids, 0] = strength * angle.cos()
        self.force[ids, 1] = strength * angle.sin()
        self.remaining[ids] = self.uniform(ids, self.cfg.push_duration_s)
        self.cooldown[ids] = self.remaining[ids] + self.uniform(ids, self.cfg.push_interval_s)
        self.push_count[ids] += 1
        self.peak_force[ids] = torch.maximum(self.peak_force[ids], strength)
        self.challenge_started |= eligible & test
        self.challenge_active |= eligible & test
        stopped = ~live | self.clean
        self.force[stopped] = 0
        self.remaining[stopped] = 0
        self.write_force(slice(None))

    def metrics(self):
        return {
            'gravity_scale': self.gravity_scale,
            'clean_episode': self.clean,
            'push_count': self.push_count,
            'peak_push_force': self.peak_force,
            'hold_push_completed': self.challenge_done,
        }
