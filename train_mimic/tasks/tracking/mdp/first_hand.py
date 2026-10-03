"""Residual reference tracking for one contact-gated left-hand ladder transfer."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch
import torch.nn.functional as F
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from train_mimic.data.first_hand_reference import DEFAULT_FIRST_HAND_REFERENCE, load_first_hand_reference
from .ladder import LadderClimbCommand, LadderClimbCommandCfg, LadderPhase, _matrix_to_quaternion


class FirstHandStage(IntEnum):
    STABILIZE = 0
    PREPARE = 1
    RELEASE = 2
    TRANSFER = 3
    HOLD = 4
    DONE = 5


class FirstHandFailure(IntEnum):
    NONE = 0
    RIGHT_HAND_LOST = 1
    SUPPORT_LOST = 2
    RELEASE_STALLED = 3
    STAGE_TIMEOUT = 4


@dataclass(kw_only=True)
class FirstHandCommandCfg(LadderClimbCommandCfg):
    reference_file: str = DEFAULT_FIRST_HAND_REFERENCE
    reference_robot_xml: str | None = None
    motion_time_scale: float = 1.0
    endpoint_tolerance: float = 0.04
    stable_hold_s: float = 0.30
    minimum_hold_s: float = 0.50
    support_loss_grace_s: float = 0.15
    stabilization_timeout_s: float = 5.0
    preparation_timeout_s: float = 3.0
    transfer_wait_timeout_s: float = 3.0
    hold_timeout_s: float = 3.0
    disturbances_enabled: bool = False
    gravity_scale_range: tuple[float, float] = (.97, 1.03)
    clean_episode_probability: float = .2
    push_weight_fraction: tuple[float, float] = (.02, .04)
    push_duration_s: tuple[float, float] = (.1, .2)
    push_interval_s: tuple[float, float] = (2., 4.)
    hold_push_delay_s: float = .5

    def build(self, env):
        return FirstHandCommand(self, env)


class FirstHandCommand(LadderClimbCommand):
    cfg: FirstHandCommandCfg

    def __init__(self, cfg, env):
        if cfg.first_moving_hand != "left" or cfg.curriculum_enabled or cfg.boundary_state_reset_prob:
            raise ValueError("First-hand tracking requires left hand, fixed curriculum and fresh resets")
        for name in ("motion_time_scale", "endpoint_tolerance", "stable_hold_s", "minimum_hold_s",
                     "support_loss_grace_s", "stabilization_timeout_s", "preparation_timeout_s",
                     "transfer_wait_timeout_s", "hold_timeout_s"):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        super().__init__(cfg, env)
        self.disturbances = None
        if cfg.disturbances_enabled:
            for name in ('gravity_scale_range', 'push_weight_fraction', 'push_duration_s', 'push_interval_s'):
                lo, hi = getattr(cfg, name)
                if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
                    raise ValueError(f'{name} must be positive finite ordered bounds')
            if not 0 <= cfg.clean_episode_probability <= 1:
                raise ValueError('clean_episode_probability must be between zero and one')
            if not math.isfinite(cfg.hold_push_delay_s) or cfg.hold_push_delay_s < 0:
                raise ValueError('hold_push_delay_s must be finite and nonnegative')
            if cfg.hold_timeout_s <= max(cfg.minimum_hold_s, cfg.hold_push_delay_s + cfg.push_duration_s[1] + cfg.stable_hold_s):
                raise ValueError('hold_timeout_s must allow the test pulse and recovery window')
            from .first_hand_disturbances import FirstHandDisturbances
            self.disturbances = FirstHandDisturbances(self)
        reference = load_first_hand_reference(cfg.reference_file, cfg.reference_robot_xml)
        self.reference_signature = reference["signature"]
        joint_order = [reference["joint_names"].index(n) for n in self.robot.joint_names]
        self.ref_fps = reference["fps"]
        self._joint_reference = torch.tensor(reference["joint_pos"][:, joint_order], device=self.device)
        self._velocity_reference = torch.tensor(reference["joint_vel"][:, joint_order], device=self.device)
        self._position_reference = torch.tensor(reference["position"], device=self.device)
        self._quaternion_reference = torch.tensor(reference["quaternion"], device=self.device)
        self.motion_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reference_time = torch.zeros(self.num_envs, device=self.device)
        self.stage_elapsed = torch.zeros_like(self.reference_time)
        self.hold_elapsed = torch.zeros_like(self.reference_time)
        self.support_loss_elapsed = torch.zeros_like(self.reference_time)
        self.motion_failed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.failure_reason = torch.zeros_like(self.motion_stage)
        self.failure_stage = torch.full_like(self.motion_stage, -1)
        self.max_motion_stage = torch.zeros_like(self.motion_stage)
        self.progress_pulse = torch.zeros_like(self.reference_time)
        self.release_pulse = torch.zeros_like(self.motion_failed)
        self.grasp_pulse = torch.zeros_like(self.motion_failed)
        self.success_pulse = torch.zeros_like(self.motion_failed)
        self.best_reach = torch.zeros_like(self.reference_time)
        self._hold_hand_offset = torch.zeros((self.num_envs, 3), device=self.device)
        self._compute_dt = 0.0
        self.reference_joint_pos = self._joint_reference[0].expand(self.num_envs, -1).clone()
        self.reference_joint_vel = torch.zeros_like(self.reference_joint_pos)
        self.reference_position = self._position_reference[0].expand(self.num_envs, -1, -1).clone()
        self.reference_quaternion = self._quaternion_reference[0].expand(self.num_envs, -1, -1).clone()
        for name in ("motion_stage", "reference_time", "hand_tracking_error", "hold_progress",
                     "motion_failed", "first_hand_success", "max_motion_stage"):
            self.metrics[name] = torch.zeros_like(self.reference_time)
        if self.disturbances is not None:
            for name in self.disturbances.metrics():
                self.metrics[name] = torch.zeros_like(self.reference_time)
        for stage in FirstHandStage:
            if stage == FirstHandStage.DONE:
                continue
            for reason in FirstHandFailure:
                if reason != FirstHandFailure.NONE:
                    self.metrics[f"failure/{stage.name.lower()}/{reason.name.lower()}"] = torch.zeros_like(self.reference_time)

    def compute(self, dt):
        # CommandTerm.compute logs BEFORE advancing the command. Here a terminal
        # flag must be logged AFTER the transition, before next step's auto-reset.
        # Preserve its resampling/timer behavior, changing only the metric order.
        self._compute_dt = dt
        self.time_left -= dt
        resample_ids = (self.time_left <= 0.).nonzero().flatten()
        if len(resample_ids):
            self._resample(resample_ids)
        self._update_command()
        self._update_metrics()

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.motion_stage[env_ids] = int(FirstHandStage.STABILIZE)
        self.failure_stage[env_ids] = -1
        for buffer in (self.reference_time, self.stage_elapsed, self.hold_elapsed,
                       self.support_loss_elapsed, self.motion_failed, self.progress_pulse,
                       self.release_pulse, self.grasp_pulse, self.success_pulse, self.best_reach, self._hold_hand_offset,
                       self.failure_reason, self.max_motion_stage):
            buffer[env_ids] = 0
        self.reference_joint_pos[env_ids] = self._joint_reference[0]
        self.reference_joint_vel[env_ids] = 0
        self.reference_position[env_ids] = self._position_reference[0]
        self.reference_quaternion[env_ids] = self._quaternion_reference[0]
        if self.disturbances is not None:
            self.disturbances.reset(env_ids)

    def _set_stage(self, env_ids, stage):
        self.motion_stage[env_ids] = int(stage)
        self.max_motion_stage[env_ids] = torch.maximum(self.max_motion_stage[env_ids], self.motion_stage[env_ids])
        self.stage_elapsed[env_ids] = 0
        self._phase_dwell_count[env_ids] = 0

    def _fail(self, mask, reason):
        # Keep the first cause/stage for the rest of the episode. Call order gives
        # a deterministic primary cause if several conditions fail simultaneously.
        selected = mask & ~self.motion_failed & ~self.finished
        self.failure_reason[selected] = int(reason)
        self.failure_stage[selected] = self.motion_stage[selected]
        self.motion_failed[selected] = True

    def _update_command(self):
        self._stage_at_start = self.motion_stage.clone()
        self._fresh_reset = self._pending_start_pose_init.clone()
        self._dt = torch.where(self._fresh_reset, 0., self._compute_dt)
        self.stage_elapsed += self._dt * (~self.finished & ~self.motion_failed)
        self.progress_pulse.zero_()
        self.release_pulse.zero_()
        self.grasp_pulse.zero_()
        self.success_pulse.zero_()
        previous_time = self.reference_time.clone()
        super()._update_command()
        # A vanished stationary hand weld is a failure; foot contact gets a short grace.
        live = self.initialized & ~self.finished & ~self._fresh_reset
        required = self.required_supports
        self.support_loss_elapsed[:] = torch.where(required, 0., self.support_loss_elapsed + self._dt)
        self._fail(live & ~self.attached[self._all_env_ids, 1-self.active_hand], FirstHandFailure.RIGHT_HAND_LOST)
        self._fail(live & (self.support_loss_elapsed >= self.cfg.support_loss_grace_s), FirstHandFailure.SUPPORT_LOST)
        limits = torch.tensor([
            self.cfg.stabilization_timeout_s, self.cfg.preparation_timeout_s*self.cfg.motion_time_scale,
            self.cfg.pre_release_timeout_steps*self._env.step_dt,
            .04 + 2.2*self.cfg.motion_time_scale + self.cfg.transfer_wait_timeout_s,
            self.cfg.hold_timeout_s, 1e9,
        ], device=self.device)
        self._fail(self.pre_release_stalled, FirstHandFailure.RELEASE_STALLED)
        stage_limit = torch.where(self.motion_stage == int(FirstHandStage.TRANSFER),
                                  .04 + (self.reference_end_time-self.reference_transfer_start)*self.cfg.motion_time_scale + self.cfg.transfer_wait_timeout_s,
                                  limits[self.motion_stage])
        self._fail(self.stage_elapsed > stage_limit, FirstHandFailure.STAGE_TIMEOUT)
        # A detach that loses the remaining supports is not a successful release.
        self.release_pulse &= ~self.motion_failed & self.required_supports
        distance = torch.linalg.vector_norm(self.active_hand_pos_w - self.endpoint_position, dim=-1)
        initial_distance = torch.linalg.vector_norm(self.endpoint_position - self.initial_hand_position, dim=-1)
        reach = (1 - distance/initial_distance.clamp_min(1e-6)).clamp(0, 1)
        valid = (self.motion_stage >= int(FirstHandStage.TRANSFER)) & required & ~self.motion_failed
        record = torch.where(valid, torch.maximum(self.best_reach, reach), self.best_reach)
        self.progress_pulse[:] = record - self.best_reach
        self.best_reach[:] = record
        rate = torch.where(self._dt > 0, (self.reference_time-previous_time)/self._dt.clamp_min(1e-6), 0.)
        # Phase-boundary jumps traverse only stationary portions of the reference.
        self._sample_reference(rate.clamp(0, 1/self.cfg.motion_time_scale))
        if self.disturbances is not None:
            self.disturbances.tick(self._dt)

    @property
    def required_supports(self):
        require_moving_hand = (self.motion_stage <= int(FirstHandStage.RELEASE)) | (self.motion_stage >= int(FirstHandStage.HOLD))
        return self.attached[self._all_env_ids, 1-self.active_hand] & self.foot_support.all(dim=1) & (~require_moving_hand | self.attached[self._all_env_ids, self.active_hand])

    def _advance_stabilization_phase(self, phase_at_start):
        del phase_at_start
        mask = (self._stage_at_start == int(FirstHandStage.STABILIZE)) & ~self._fresh_reset & ~self.motion_failed
        self._update_phase_dwell(mask, self.stabilization_conditions_satisfied)
        ids = torch.where(mask & (self._phase_dwell_count >= self._stabilization_dwell_target))[0]
        self._set_stage(ids, FirstHandStage.PREPARE)
        self.phase[ids] = int(LadderPhase.FIRST_HAND)
        self.target_rung[ids] = self.cfg.start_rung + 1
        self.reference_time[ids] = 1.0
        self.just_stabilized[ids] = True

    def _advance_hand_phase(self, phase_at_start):
        del phase_at_start
        live = ~self.motion_failed & ~self.finished & ~self._fresh_reset
        prepare = live & (self._stage_at_start == int(FirstHandStage.PREPARE))
        self.reference_time[prepare] = (self.reference_time[prepare]
                                        + self._dt[prepare]/self.cfg.motion_time_scale)
        self.reference_time[prepare] = torch.minimum(self.reference_time[prepare], self.reference_prepare_end[prepare])
        ready = prepare & (self.reference_time >= self.reference_prepare_end-1e-6) & self._release_is_stable()
        ids = torch.where(ready)[0]
        self._set_stage(ids, FirstHandStage.RELEASE)
        for hand in (0, 1):
            self._begin_release(ids[self.active_hand[ids] == hand], hand)

        release = live & (self._stage_at_start == int(FirstHandStage.RELEASE))
        self._advance_release_ramp(release)
        ids = torch.where(release & ~self.attached[self._all_env_ids, self.active_hand] & ~self._release_active)[0]
        self._set_stage(ids, FirstHandStage.TRANSFER)
        self.release_pulse[ids] = True
        self.reference_time[ids] = self.reference_transfer_start[ids]

        transfer = live & (self._stage_at_start == int(FirstHandStage.TRANSFER))
        move = transfer & (self.stage_elapsed >= .04)
        self.reference_time[move] = (self.reference_time[move]
                                     + self._dt[move]/self.cfg.motion_time_scale)
        self.reference_time[move] = torch.minimum(self.reference_time[move], self.reference_end_time[move])
        endpoint_error = torch.linalg.vector_norm(self.active_hand_pos_w - self.endpoint_position, dim=-1)
        rung_error = torch.linalg.vector_norm(self.active_hand_pos_w-self.target_pos_w, dim=-1)
        speed = torch.linalg.vector_norm(self.active_hand_vel_w, dim=-1)
        reached = (self.reference_time >= self.reference_end_time-1e-6) & ~self.attached[self._all_env_ids, self.active_hand] & self.required_supports
        reached &= (endpoint_error <= self.cfg.endpoint_tolerance) & (rung_error <= self.cfg.attach_distance)
        reached &= (speed <= self.cfg.max_attach_speed) & self.phase_completion_stable
        reached &= self.torso_orientation_error <= self.cfg.max_phase_torso_orientation_error
        self._update_phase_dwell(transfer, reached)
        ids = torch.where(transfer & (self._phase_dwell_count >= self.cfg.hand_target_dwell_steps))[0]
        for hand in (0, 1):
            selected = ids[self.active_hand[ids] == hand]
            self._attach(selected, hand, self.target_rung[selected])
        self._hold_hand_offset[ids] = self.active_hand_pos_w[ids] - self.endpoint_position[ids]
        self.grasp_pulse[ids] = True
        self.just_advanced[ids] = True
        self._set_stage(ids, FirstHandStage.HOLD)

        hold = live & (self._stage_at_start == int(FirstHandStage.HOLD))
        stable = self.required_supports & self.phase_completion_stable
        stable &= self.torso_orientation_error <= self.cfg.max_phase_torso_orientation_error
        stable &= self.held_rung[self._all_env_ids, self.active_hand] == self.cfg.start_rung + 1
        if self.disturbances is not None:
            stable &= self.disturbances.recovery_ready
            stable &= torch.linalg.vector_norm(self.torso_ang_vel_w, dim=-1) <= self.cfg.max_stabilization_body_angular_speed
            stable &= torch.linalg.vector_norm(self.pelvis_ang_vel_w, dim=-1) <= self.cfg.max_stabilization_body_angular_speed
            stable &= self.robot.data.joint_vel[:, self._waist_joint_ids].abs().amax(dim=-1) <= self.cfg.max_stabilization_waist_joint_speed
        self.hold_elapsed[hold] = torch.where(stable[hold], self.hold_elapsed[hold]+self._dt[hold], 0.)
        ready = hold & (self.hold_elapsed >= self.cfg.stable_hold_s) & (self.stage_elapsed >= self.cfg.minimum_hold_s)
        ids = torch.where(ready)[0]
        self._complete_motion(ids)

    def _complete_motion(self, ids):
        self.finished[ids] = True
        self.success_pulse[ids] = True
        self._set_stage(ids, FirstHandStage.DONE)

    def _advance_foot_phase(self, phase_at_start):
        del phase_at_start  # Authored hand-motion tasks do not transfer the feet.

    def sample(self, table, times):
        index = (times*self.ref_fps).clamp(0, len(table)-1)
        low = index.long()
        high = (low+1).clamp(max=len(table)-1)
        alpha = (index-low).reshape((-1,) + (1,)*(table.ndim-1))
        return torch.lerp(table[low], table[high], alpha)

    @property
    def reference_prepare_end(self):
        return torch.full_like(self.reference_time, 1.7)

    @property
    def reference_transfer_start(self):
        return torch.full_like(self.reference_time, 1.7)

    @property
    def reference_end_time(self):
        return torch.full_like(self.reference_time, 3.9)

    @property
    def endpoint_position(self):
        return self._position_reference[-1,0].expand(self.num_envs,-1)

    @property
    def initial_hand_position(self):
        return self._position_reference[0,0].expand(self.num_envs,-1)

    def future_joint_reference(self, times):
        return self.sample(self._joint_reference, times)

    def _sample_reference(self, rate=None):
        self.reference_joint_pos[:] = self.sample(self._joint_reference, self.reference_time)
        self.reference_joint_vel[:] = self.sample(self._velocity_reference, self.reference_time)
        self.reference_joint_vel *= (torch.zeros_like(self.reference_time) if rate is None else rate)[:, None]
        self.reference_position[:] = self.sample(self._position_reference, self.reference_time)
        self.reference_position[:, 0] += self._hold_hand_offset
        self.reference_quaternion[:] = F.normalize(self.sample(self._quaternion_reference, self.reference_time), dim=-1)

    @property
    def actual_position(self):
        return torch.cat((self.hand_pos_w, self.foot_pos_w,
                          self._env.sim.data.xpos[:, [self._pelvis_body_id, self._torso_body_id]]), dim=1)

    @property
    def actual_quaternion(self):
        sites = self._env.sim.data.site_xmat[:, torch.cat((self._hand_site_ids, self._foot_site_ids))]
        bodies = self._env.sim.data.xmat[:, [self._pelvis_body_id, self._torso_body_id]]
        return _matrix_to_quaternion(torch.cat((sites.reshape(self.num_envs, 4, 3, 3),
                                                bodies.reshape(self.num_envs, 2, 3, 3)), dim=1))

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["motion_stage"][:] = self.motion_stage
        self.metrics["reference_time"][:] = self.reference_time
        self.metrics["hand_tracking_error"][:] = torch.linalg.vector_norm(self.active_hand_pos_w-self.reference_position[self._all_env_ids,self.active_hand], dim=-1)
        self.metrics["hold_progress"][:] = self.hold_elapsed/self.cfg.stable_hold_s
        self.metrics["motion_failed"][:] = self.motion_failed
        self.metrics["first_hand_success"][:] = self.finished
        self.metrics["max_motion_stage"][:] = self.max_motion_stage
        if self.disturbances is not None:
            for name, value in self.disturbances.metrics().items():
                self.metrics[name][:] = value
        for stage in FirstHandStage:
            if stage == FirstHandStage.DONE:
                continue
            for reason in FirstHandFailure:
                if reason != FirstHandFailure.NONE:
                    self.metrics[f"failure/{stage.name.lower()}/{reason.name.lower()}"][:] = (
                        self.motion_failed & (self.failure_stage == int(stage)) & (self.failure_reason == int(reason))
                    )


@dataclass(kw_only=True)
class ResidualReferenceActionCfg(JointPositionActionCfg):
    command_name: str = "ladder"
    use_default_offset: bool = False
    residual_bound: float = 1.0

    def build(self, env):
        return ResidualReferenceAction(self, env)


class ResidualReferenceAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isfinite(cfg.residual_bound) or cfg.residual_bound <= 0:
            raise ValueError("residual_bound must be positive")
        self._reference = env.command_manager.get_term(cfg.command_name)

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        delta = actions.clamp(-self.cfg.residual_bound, self.cfg.residual_bound)*self._scale
        target = self._reference.reference_joint_pos[:, self._target_ids] + delta
        limits = self._entity.data.joint_pos_limits[:, self._target_ids]
        self._processed_actions[:] = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])


def command(env, command_name="ladder") -> FirstHandCommand:
    return env.command_manager.get_term(command_name)


def reference_observation(env, command_name="ladder"):
    c = command(env, command_name)
    position_error = c._vectors_to_torso((c.reference_position-c.actual_position).reshape(env.num_envs, -1, 3))
    future_time = torch.minimum(c.reference_time + .2/c.cfg.motion_time_scale,
                               c.reference_end_time)
    future = c.future_joint_reference(future_time)
    return torch.cat(((c.reference_time/c.reference_end_time)[:, None],
                      c.reference_joint_pos-c.robot.data.joint_pos,
                      c.reference_joint_vel,
                      future-c.robot.data.joint_pos,
                      position_error.flatten(1), c.attached.float(), c.foot_support.float()), dim=1)


def stage_observation(env, command_name="ladder"):
    return F.one_hot(command(env, command_name).motion_stage, len(FirstHandStage)).float()


def joint_tracking_cost(env, command_name="ladder", std=.2):
    c = command(env, command_name)
    return ((c.robot.data.joint_pos-c.reference_joint_pos)/std).square().clamp(max=9).mean(dim=1)


def joint_velocity_tracking_cost(env, command_name="ladder", std=2.):
    c = command(env, command_name)
    return ((c.robot.data.joint_vel-c.reference_joint_vel)/std).square().clamp(max=9).mean(dim=1)


def body_angular_velocity_cost(env, command_name="ladder", body="torso", std=.4):
    """Quadratic angular-speed cost; one unit at the stabilization threshold."""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be positive and finite")
    c = command(env, command_name)
    if body == "torso":
        velocity = c.torso_ang_vel_w
    elif body == "pelvis":
        velocity = c.pelvis_ang_vel_w
    else:
        raise ValueError("body must be torso or pelvis")
    return velocity.square().sum(dim=-1)/std**2


def waist_velocity_cost(env, command_name="ladder", std=.6):
    """Penalize the fastest waist joint, matching the stabilization gate."""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be positive and finite")
    c = command(env, command_name)
    return c.robot.data.joint_vel[:, c._waist_joint_ids].square().amax(dim=-1)/std**2


def position_tracking_cost(env, command_name="ladder", track_ids=(0,), std=.04):
    c = command(env, command_name)
    error = c.actual_position[:, track_ids]-c.reference_position[:, track_ids]
    return (error.square().sum(dim=-1)/std**2).clamp(max=9).mean(dim=1)


def orientation_tracking_cost(env, command_name="ladder", track_ids=(0, 4, 5), std=.2):
    c = command(env, command_name)
    error = quat_error_magnitude(c.actual_quaternion[:, track_ids], c.reference_quaternion[:, track_ids])
    return (error/std).square().clamp(max=9).mean(dim=1)


def release_orientation_cost(env, command_name="ladder", margin=.05, std=.05):
    """Penalize torso error above a safe margin inside the actual release gate."""
    c = command(env, command_name)
    limit = c.cfg.max_release_torso_orientation_error
    if not math.isfinite(margin) or not 0 <= margin < limit:
        raise ValueError("margin must be finite and inside the release orientation limit")
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be positive and finite")
    active = ((c.motion_stage == int(FirstHandStage.PREPARE))
              | (c.motion_stage == int(FirstHandStage.RELEASE)))
    excess = (c.torso_orientation_error - (limit-margin)).clamp_min(0.)
    return (excess/std).square() * active.float()


def proximity_cost(joint_pos, limits, margin_fraction=.15):
    """Zero in the interior; quadratic within a margin of each *hard* limit.

    margin_fraction is a fraction of the full joint range at each end. Unlimited
    or degenerate ranges are ignored; outside the hard limit the cost exceeds 1.
    """
    if not 0 < margin_fraction < .5:
        raise ValueError("margin_fraction must be between zero and 0.5")
    lower, upper = limits[..., 0], limits[..., 1]
    span = upper-lower
    valid = torch.isfinite(lower) & torch.isfinite(upper) & (span > 1e-6)
    distance = torch.minimum(joint_pos-lower, upper-joint_pos)
    margin = torch.where(valid, span*margin_fraction, torch.ones_like(span))
    depth = (1-distance/margin).clamp_min(0)
    return torch.where(valid, depth.square(), 0.).sum(dim=-1)/valid.sum(dim=-1).clamp_min(1)


def joint_limit_proximity(env, command_name="ladder", margin_fraction=.15):
    c = command(env, command_name)
    return proximity_cost(c.robot.data.joint_pos, c.robot.data.joint_pos_limits, margin_fraction)


def support_reward(env, command_name="ladder"):
    return command(env, command_name).required_supports.float()


def missing_support_cost(env, command_name="ladder"):
    return (~command(env, command_name).required_supports).float()


def event_reward(env, command_name="ladder", event="progress"):
    c = command(env, command_name)
    value = {"progress": c.progress_pulse, "release": c.release_pulse,
             "grasp": c.grasp_pulse, "success": c.success_pulse}[event]
    return value.float()/env.step_dt


def failure_penalty(env, command_name="ladder"):
    return (env.termination_manager.dones & ~command(env, command_name).finished).float()/env.step_dt


def first_hand_failed(env, command_name="ladder"):
    return command(env, command_name).motion_failed
