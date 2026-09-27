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
        self.progress_pulse = torch.zeros_like(self.reference_time)
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
                     "motion_failed", "first_hand_success"):
            self.metrics[name] = torch.zeros_like(self.reference_time)

    def compute(self, dt):
        self._compute_dt = dt
        super().compute(dt)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.motion_stage[env_ids] = int(FirstHandStage.STABILIZE)
        for buffer in (self.reference_time, self.stage_elapsed, self.hold_elapsed,
                       self.support_loss_elapsed, self.motion_failed, self.progress_pulse,
                       self.grasp_pulse, self.success_pulse, self.best_reach, self._hold_hand_offset):
            buffer[env_ids] = 0
        self.reference_joint_pos[env_ids] = self._joint_reference[0]
        self.reference_joint_vel[env_ids] = 0
        self.reference_position[env_ids] = self._position_reference[0]
        self.reference_quaternion[env_ids] = self._quaternion_reference[0]

    def _set_stage(self, env_ids, stage):
        self.motion_stage[env_ids] = int(stage)
        self.stage_elapsed[env_ids] = 0
        self._phase_dwell_count[env_ids] = 0

    def _update_command(self):
        self._stage_at_start = self.motion_stage.clone()
        self._fresh_reset = self._pending_start_pose_init.clone()
        self._dt = torch.where(self._fresh_reset, 0., self._compute_dt)
        self.stage_elapsed += self._dt * (~self.finished & ~self.motion_failed)
        self.progress_pulse.zero_()
        self.grasp_pulse.zero_()
        self.success_pulse.zero_()
        previous_time = self.reference_time.clone()
        super()._update_command()
        # A vanished stationary hand weld is a failure; foot contact gets a short grace.
        live = self.initialized & ~self.finished & ~self._fresh_reset
        required = self.required_supports
        self.support_loss_elapsed[:] = torch.where(required, 0., self.support_loss_elapsed + self._dt)
        self.motion_failed |= live & (self.support_loss_elapsed >= self.cfg.support_loss_grace_s)
        self.motion_failed |= live & ~self.attached[:, 1]
        limits = torch.tensor([
            self.cfg.stabilization_timeout_s, self.cfg.preparation_timeout_s*self.cfg.motion_time_scale,
            self.cfg.pre_release_timeout_steps*self._env.step_dt,
            .04 + 2.2*self.cfg.motion_time_scale + self.cfg.transfer_wait_timeout_s,
            self.cfg.hold_timeout_s, 1e9,
        ], device=self.device)
        self.motion_failed |= ~self.finished & (self.stage_elapsed > limits[self.motion_stage])
        self.motion_failed |= self.pre_release_stalled
        distance = torch.linalg.vector_norm(self.hand_pos_w[:, 0] - self._position_reference[-1, 0], dim=-1)
        initial_distance = torch.linalg.vector_norm(self._position_reference[-1, 0] - self._position_reference[0, 0])
        reach = (1 - distance/initial_distance.clamp_min(1e-6)).clamp(0, 1)
        valid = (self.motion_stage >= int(FirstHandStage.TRANSFER)) & required & ~self.motion_failed
        record = torch.where(valid, torch.maximum(self.best_reach, reach), self.best_reach)
        self.progress_pulse[:] = record - self.best_reach
        self.best_reach[:] = record
        rate = torch.where(self._dt > 0, (self.reference_time-previous_time)/self._dt.clamp_min(1e-6), 0.)
        # Phase-boundary jumps traverse only stationary portions of the reference.
        self._sample_reference(rate.clamp(0, 1/self.cfg.motion_time_scale))

    @property
    def required_supports(self):
        require_left = (self.motion_stage <= int(FirstHandStage.RELEASE)) | (self.motion_stage >= int(FirstHandStage.HOLD))
        return self.attached[:, 1] & self.foot_support.all(dim=1) & (~require_left | self.attached[:, 0])

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
                                        + self._dt[prepare]/self.cfg.motion_time_scale).clamp(max=1.7)
        ready = prepare & (self.reference_time >= 1.7-1e-6) & self._release_is_stable()
        ids = torch.where(ready)[0]
        self._set_stage(ids, FirstHandStage.RELEASE)
        self._begin_release(ids, 0)

        release = live & (self._stage_at_start == int(FirstHandStage.RELEASE))
        self._advance_release_ramp(release)
        ids = torch.where(release & ~self.attached[:, 0] & ~self._release_active)[0]
        self._set_stage(ids, FirstHandStage.TRANSFER)

        transfer = live & (self._stage_at_start == int(FirstHandStage.TRANSFER))
        move = transfer & (self.stage_elapsed >= .04)
        self.reference_time[move] = (self.reference_time[move]
                                     + self._dt[move]/self.cfg.motion_time_scale).clamp(max=3.9)
        endpoint_error = torch.linalg.vector_norm(self.hand_pos_w[:, 0] - self._position_reference[-1, 0], dim=-1)
        rung_error = torch.linalg.vector_norm(self.active_hand_pos_w-self.target_pos_w, dim=-1)
        speed = torch.linalg.vector_norm(self.active_hand_vel_w, dim=-1)
        reached = (self.reference_time >= 3.9-1e-6) & ~self.attached[:, 0] & self.required_supports
        reached &= (endpoint_error <= self.cfg.endpoint_tolerance) & (rung_error <= self.cfg.attach_distance)
        reached &= (speed <= self.cfg.max_attach_speed) & self.phase_completion_stable
        reached &= self.torso_orientation_error <= self.cfg.max_phase_torso_orientation_error
        self._update_phase_dwell(transfer, reached)
        ids = torch.where(transfer & (self._phase_dwell_count >= self.cfg.hand_target_dwell_steps))[0]
        self._attach(ids, 0, self.target_rung[ids])
        self._hold_hand_offset[ids] = self.hand_pos_w[ids, 0] - self._position_reference[-1, 0]
        self.grasp_pulse[ids] = True
        self.just_advanced[ids] = True
        self._set_stage(ids, FirstHandStage.HOLD)

        hold = live & (self._stage_at_start == int(FirstHandStage.HOLD))
        stable = self.required_supports & self.phase_completion_stable
        stable &= self.torso_orientation_error <= self.cfg.max_phase_torso_orientation_error
        stable &= self.held_rung[:, 0] == self.cfg.start_rung + 1
        self.hold_elapsed[hold] = torch.where(stable[hold], self.hold_elapsed[hold]+self._dt[hold], 0.)
        ready = hold & (self.hold_elapsed >= self.cfg.stable_hold_s) & (self.stage_elapsed >= self.cfg.minimum_hold_s)
        ids = torch.where(ready)[0]
        self.finished[ids] = True
        self.success_pulse[ids] = True
        self._set_stage(ids, FirstHandStage.DONE)

    def _advance_foot_phase(self, phase_at_start):
        del phase_at_start  # This task never transitions to the second hand or feet.

    def sample(self, table, times):
        index = (times*self.ref_fps).clamp(0, len(table)-1)
        low = index.long()
        high = (low+1).clamp(max=len(table)-1)
        alpha = (index-low).reshape((-1,) + (1,)*(table.ndim-1))
        return torch.lerp(table[low], table[high], alpha)

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
        self.metrics["hand_tracking_error"][:] = torch.linalg.vector_norm(self.hand_pos_w[:, 0]-self.reference_position[:, 0], dim=-1)
        self.metrics["hold_progress"][:] = self.hold_elapsed/self.cfg.stable_hold_s
        self.metrics["motion_failed"][:] = self.motion_failed
        self.metrics["first_hand_success"][:] = self.finished


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
                               torch.full_like(c.reference_time, 3.9))
    future = c.sample(c._joint_reference, future_time)
    return torch.cat(((c.reference_time/3.9)[:, None],
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


def position_tracking_cost(env, command_name="ladder", track_ids=(0,), std=.04):
    c = command(env, command_name)
    error = c.actual_position[:, track_ids]-c.reference_position[:, track_ids]
    return (error.square().sum(dim=-1)/std**2).clamp(max=9).mean(dim=1)


def orientation_tracking_cost(env, command_name="ladder", track_ids=(0, 4, 5), std=.2):
    c = command(env, command_name)
    error = quat_error_magnitude(c.actual_quaternion[:, track_ids], c.reference_quaternion[:, track_ids])
    return (error/std).square().clamp(max=9).mean(dim=1)


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
    value = {"progress": c.progress_pulse, "grasp": c.grasp_pulse, "success": c.success_pulse}[event]
    return value.float()/env.step_dt


def failure_penalty(env, command_name="ladder"):
    return (env.termination_manager.dones & ~command(env, command_name).finished).float()/env.step_dt


def first_hand_failed(env, command_name="ladder"):
    return command(env, command_name).motion_failed
