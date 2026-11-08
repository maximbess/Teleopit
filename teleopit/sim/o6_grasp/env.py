"""CPU MuJoCo environment with a Gym-style API and no Gym/training dependency."""
import mujoco
import numpy as np

from .config import GraspConfig
from .controller import Controller, Mode, phase, validate_action
from .diagnostics import Verification, contact_diagnostics, relative_palm, reward_terms
from .observations import VERSION, ObservationSensor, build_observation
from .scene import build_scene, model_definition
from .wrenches import apply_wrenches


class O6GraspEnv:
    action_shape = (6,)
    observation_shape = (48,)

    def __init__(self, config=None):
        self.config = config or GraspConfig()
        self.model, self.scene_xml = build_scene(self.config)
        self.data = mujoco.MjData(self.model)
        _, active, passive = model_definition()
        self.active_names = tuple(j.get('name') for j in active)
        joints = [self.model.joint(name).id for name in self.active_names]
        self.active_qpos = self.model.jnt_qposadr[joints]
        self.active_dofs = self.model.jnt_dofadr[joints]
        self.limits = self.model.jnt_range[joints].copy()
        self.couplings = [(self.model.joint(j.get('name')).qposadr[0],
                           self.model.joint(j.find('mimic').get('joint')).qposadr[0],
                           float(j.find('mimic').get('multiplier')),
                           float(j.find('mimic').get('offset'))) for j in passive]
        self.palm_body = self.model.body('rh_hand_base_link').id
        self.palm_site = self.model.site('palm_frame').id
        self.rung_geom = self.model.geom('rung_collision').id if self.config.rung_enabled else -1
        self.reset()

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('configure scene using GraspConfig, not reset options')
        cfg = self.config
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3], self.data.qpos[3:7] = cfg.palm_pos, cfg.palm_quat
        self.data.qpos[self.active_qpos] = cfg.open_pose
        for passive, active, ratio, offset in self.couplings:
            self.data.qpos[passive] = offset + ratio * self.data.qpos[active]
        self.controller = Controller(cfg, self.limits)
        self.data.ctrl[:] = cfg.open_pose
        mujoco.mj_forward(self.model, self.data)
        self.fixture_position = self.data.site('palm_frame').xpos.copy()
        self.fixture_rotation = self.data.site('palm_frame').xmat.reshape(3, 3).copy()
        self.verifier = Verification()
        self.sensor = ObservationSensor(cfg, seed)
        self.steps = 0
        self.done = False
        self.last_nominal = np.asarray(cfg.open_pose)
        self.last_fixture = np.zeros(6)
        self.last_external = np.zeros(6)
        self.last_gain = 1.
        self.interval_unsupported = False
        return self._observe(Mode.OPEN, 0., 0.)

    def palm_velocity(self, *, local):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                self.palm_site, velocity, int(local))
        return velocity

    def step(self, action):
        action = validate_action(action)
        if self.done:
            raise RuntimeError('episode finished; call reset()')
        cfg = self.config
        previous = self.controller.previous_action.copy()
        time = self.steps * cfg.control_dt
        target, nominal, mode, _, _ = self.controller.command(time, action)
        self.data.ctrl[:] = target
        self.last_nominal = nominal
        self.interval_unsupported = True
        for _ in range(round(cfg.control_dt / cfg.physics_dt)):
            submode, elapsed, _ = phase(self.data.time, cfg.phase_durations)
            fixture, external, gain = apply_wrenches(self, submode, elapsed)
            self.interval_unsupported &= submode == Mode.HOLD and gain == 0.
            self.last_fixture, self.last_external, self.last_gain = fixture, external, gain
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        _, _, progress = phase(time, cfg.phase_durations)
        observation, info = self._observe(mode, progress, float(np.sum((self.controller.previous_action - previous) ** 2)))
        self.done = self.steps * cfg.control_dt >= sum(cfg.phase_durations) - 1e-9
        return observation, sum(info['reward_terms'].values()), False, self.done, info

    def _observe(self, mode, progress, action_change):
        raw, normalized = build_observation(self, self.last_nominal, mode, progress)
        contacts, wrench = contact_diagnostics(self)
        penetration = max((max(0., -self.data.contact[i].dist) for i in range(self.data.ncon)), default=0.)
        info = self.verifier.update(self, mode, self.interval_unsupported, contacts, penetration)
        position, rotation = relative_palm(self)
        palm = self.data.site('palm_frame')
        info.update(time=float(self.data.time), mode=mode.name, phase_progress=progress,
                    observation_version=VERSION, raw_observation=raw.tolist(),
                    normalized_observation=normalized.tolist(),
                    active_position=self.data.qpos[self.active_qpos].tolist(),
                    active_velocity=self.data.qvel[self.active_dofs].tolist(),
                    target=self.controller.target.tolist(), applied_target=self.data.ctrl.tolist(),
                    nominal=self.last_nominal.tolist(), actuator_effort=self.data.actuator_force.tolist(),
                    palm_position=palm.xpos.tolist(), palm_rotation=palm.xmat.tolist(),
                    palm_velocity=self.palm_velocity(local=False).tolist(),
                    palm_in_rung_position=position.tolist(), palm_in_rung_rotation=rotation.flatten().tolist(),
                    contacts=contacts, contact_wrench=wrench.tolist(),
                    fixture_wrench=self.last_fixture.tolist(), external_wrench=self.last_external.tolist(),
                    fixture_gain=self.last_gain,
                    fixture_stiffness=(self.last_gain * np.asarray(self.config.fixture_stiffness)).tolist(),
                    fixture_damping=(self.last_gain * np.asarray(self.config.fixture_damping)).tolist(),
                    unsupported=bool(self.interval_unsupported), max_penetration=penetration,
                    coupling_error=max(abs(self.data.qpos[p] - ratio * self.data.qpos[a] - offset)
                                       for p, a, ratio, offset in self.couplings),
                    release_clear=self._release_clear(mode, contacts),
                    numerical_warnings=[int(w.number) for w in self.data.warning])
        info['reward_terms'] = reward_terms(mode, info, action_change)
        return self.sensor.sample(normalized), info

    def _release_clear(self, mode, contacts):
        if mode != Mode.RELEASE or contacts:
            return False
        opened = np.max(np.abs(self.data.qpos[self.active_qpos] - self.config.open_pose)) < 0.08
        if not opened or self.rung_geom < 0:
            return bool(opened)
        for geom in range(self.model.ngeom):
            if self.model.geom_group[geom] != 3:
                continue
            distance = mujoco.mj_geomDistance(self.model, self.data, geom, self.rung_geom,
                                             self.config.release_clearance, None)
            if distance < self.config.release_clearance:
                return False
        return True

    def close(self):
        """No external resources are held by the headless environment."""
