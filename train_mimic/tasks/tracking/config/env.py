"""Environment builder for the General-Tracking-G1 task."""

from __future__ import annotations

import math
from copy import deepcopy
from functools import partial
from pathlib import Path

import mujoco

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils import spec_config as spec_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from train_mimic.tasks.tracking import mdp
from train_mimic.tasks.tracking.config.constants import DEFAULT_TRAIN_MOTION_FILE
from train_mimic.tasks.tracking.mdp import MotionCommandCfg
from train_mimic.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from teleopit.runtime.assets import UNITREE_G1_XML, missing_gmr_assets_message

_TRACKING_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

_TRAIN_ONLY_EVENTS = (
    "push_robot",
    "base_com",
    "add_joint_default_pos",
    "physics_material",
    "randomize_rigid_body_mass",
)

_LADDER_NUM_RUNGS = 9
_LADDER_HALF_BASE = 1.05
_LADDER_HEIGHT = 2.80
_LADDER_HALF_WIDTH = 0.35
_LADDER_RAIL_RADIUS = 0.050
_LADDER_RUNG_HALF_DEPTH = 0.055
_LADDER_RUNG_HALF_HEIGHT = 0.035
_LADDER_GRIP_RADIUS = 0.075
_LADDER_BODY_BLOCKER_HALF_THICKNESS = 0.015
_LADDER_BODY_BLOCKER_MASK = 2
_LADDER_START_RUNG = 4
_LADDER_INITIAL_FOOT_RUNG = 1
_LADDER_FOOT_CONTACT_SENSOR = "ladder_foot_contact"
_LADDER_ARM_EFFORT_SCALE = 0.70
_LADDER_CURRICULUM_MIN_PHASE_STEPS = (36_000, 120_000, 120_000, 120_000)
# Symmetric climbing pose computed against the generated left ladder face.  The
# sole sites sit just outside and above physical rung 2, while the grip sites
# are within attachment range of rung 5.
_LADDER_INITIAL_ROOT_POS = (-0.911384, 0.0, 1.357509)
_LADDER_INITIAL_JOINT_POS = {
    ".*_hip_pitch_joint": -0.228917,
    ".*_knee_joint": 0.657862,
    ".*_ankle_pitch_joint": -0.333096,
    "waist_pitch_joint": -0.184737,
    ".*_shoulder_pitch_joint": -0.070306,
    "left_shoulder_roll_joint": 0.160023,
    "right_shoulder_roll_joint": -0.160023,
    "left_shoulder_yaw_joint": -0.034580,
    "right_shoulder_yaw_joint": 0.034580,
    ".*_elbow_joint": 0.407197,
    ".*_wrist_pitch_joint": -0.097974,
}

_LADDER_FLOOR_MATERIAL = spec_cfg.MaterialCfg(
    name="ladder_floor_material",
    rgba=(0.18, 0.22, 0.26, 1.0),
    reflectance=0.0,
    geom_names_expr=("terrain$",),
)
_LADDER_SCENE_LIGHT = spec_cfg.LightCfg(
    name="ladder_sun",
    type="directional",
    pos=(-1.5, -2.0, 4.0),
    dir=(0.35, 0.2, -1.0),
    castshadow=False,
)


def _required_spec_body(spec: mujoco.MjSpec, name: str):
    for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):
        if body.name == name:
            return body
    raise ValueError(f"G1 MuJoCo XML is missing required body {name!r}")


def _add_ladder_to_g1_spec(spec: mujoco.MjSpec) -> None:
    """Augment the canonical G1 model with an A-frame ladder and grip welds."""

    existing_sites = {
        site.name for site in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_SITE)
    }
    generated_site_names = {"left_grip_site", "right_grip_site"}
    if generated_site_names & existing_sites:
        raise ValueError(
            "The ladder task requires the canonical G1 XML without pre-added grip "
            "sites. Remove custom left_grip_site/right_grip_site definitions."
        )

    left_wrist = _required_spec_body(spec, "left_wrist_yaw_link")
    right_wrist = _required_spec_body(spec, "right_wrist_yaw_link")
    left_wrist.add_site(
        name="left_grip_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=(0.10, 0.0, 0.0),
        size=(0.05,),
        group=3,
        rgba=(1.0, 0.0, 0.0, 0.7),
    )
    right_wrist.add_site(
        name="right_grip_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=(0.10, 0.0, 0.0),
        size=(0.05,),
        group=3,
        rgba=(0.0, 0.0, 1.0, 0.7),
    )

    world = spec.worldbody
    for side, base_x in (("left", -_LADDER_HALF_BASE), ("right", _LADDER_HALF_BASE)):
        side_body = world.add_body(name=f"{side}_ladder_body")
        face_length = math.hypot(base_x, _LADDER_HEIGHT)
        face_tilt = math.atan2(-base_x, _LADDER_HEIGHT)
        blocker_offset = (
            _LADDER_RUNG_HALF_DEPTH + _LADDER_BODY_BLOCKER_HALF_THICKNESS + 0.005
        )
        inward_normal_x = -math.copysign(_LADDER_HEIGHT / face_length, base_x)
        inward_normal_z = -abs(base_x) / face_length
        side_body.add_geom(
            name=f"{side}_ladder_body_blocker",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(
                0.5 * base_x + blocker_offset * inward_normal_x,
                0.0,
                0.5 * _LADDER_HEIGHT + blocker_offset * inward_normal_z,
            ),
            quat=(
                math.cos(0.5 * face_tilt),
                0.0,
                math.sin(0.5 * face_tilt),
                0.0,
            ),
            size=(
                _LADDER_BODY_BLOCKER_HALF_THICKNESS,
                _LADDER_HALF_WIDTH,
                0.5 * face_length,
            ),
            contype=_LADDER_BODY_BLOCKER_MASK,
            conaffinity=_LADDER_BODY_BLOCKER_MASK,
            condim=1,
            solref=(0.005, 1.0),
            solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
            margin=0.002,
            rgba=(0.0, 0.0, 0.0, 0.0),
        )
        for rail_index, y in enumerate(
            (-_LADDER_HALF_WIDTH, _LADDER_HALF_WIDTH),
            start=1,
        ):
            side_body.add_geom(
                name=f"{side}_ladder_rail_{rail_index:02d}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=(base_x, y, 0.0, 0.0, y, _LADDER_HEIGHT),
                size=(_LADDER_RAIL_RADIUS,),
                contype=1,
                conaffinity=1,
                condim=4,
                friction=(1.4, 0.02, 0.002),
                solref=(0.005, 1.0),
                solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
                margin=0.002,
                rgba=(0.55, 0.55, 0.58, 1.0),
            )

        for rung_index in range(1, _LADDER_NUM_RUNGS + 1):
            fraction = rung_index / (_LADDER_NUM_RUNGS + 1)
            x = base_x * (1.0 - fraction)
            z = _LADDER_HEIGHT * fraction
            fromto = (
                x,
                -_LADDER_HALF_WIDTH,
                z,
                x,
                _LADDER_HALF_WIDTH,
                z,
            )
            rung_name = f"{side}_ladder_rung_{rung_index:02d}"
            side_body.add_geom(
                name=rung_name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(x, 0.0, z),
                size=(
                    _LADDER_RUNG_HALF_DEPTH,
                    _LADDER_HALF_WIDTH,
                    _LADDER_RUNG_HALF_HEIGHT,
                ),
                contype=1,
                conaffinity=1,
                condim=4,
                friction=(1.8, 0.02, 0.002),
                solref=(0.005, 1.0),
                solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
                margin=0.002,
                rgba=(0.55, 0.55, 0.58, 1.0),
            )
            side_body.add_site(
                name=f"{rung_name}_grip",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=fromto,
                size=(_LADDER_GRIP_RADIUS,),
                group=3,
                rgba=(0.15, 0.95, 0.25, 0.25),
            )

    for hand, color in (
        ("left", (1.0, 0.5, 0.0, 0.8)),
        ("right", (0.0, 1.0, 1.0, 0.8)),
    ):
        anchor = world.add_body(
            name=f"{hand}_grip_anchor_body",
            mocap=True,
            pos=(0.0, 0.0, -1.0),
        )
        anchor.add_site(
            name=f"{hand}_grip_anchor",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.012,),
            rgba=color,
        )
        spec.add_equality(
            name=f"{hand}_grip_weld",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=f"{hand}_grip_site",
            name2=f"{hand}_grip_anchor",
            active=False,
            data=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05),
            solref=(0.03, 1.0),
            solimp=(0.90, 0.95, 0.001, 0.5, 2.0),
        )


def resolve_g1_training_xml(robot_xml: str | Path | None = None) -> Path:
    """Resolve the MuJoCo XML used for G1 policy training."""
    if robot_xml is None or str(robot_xml).strip() == "":
        return UNITREE_G1_XML.resolve()

    path = Path(robot_xml).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _get_g1_training_spec(robot_xml: str | Path | None = None) -> mujoco.MjSpec:
    xml_path = resolve_g1_training_xml(robot_xml)
    if not xml_path.is_file():
        raise FileNotFoundError(
            missing_gmr_assets_message(xml_path, label="G1 training MuJoCo XML")
        )
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def _get_g1_ladder_training_spec(
    robot_xml: str | Path | None = None,
) -> mujoco.MjSpec:
    spec = _get_g1_training_spec(robot_xml)
    _remove_embedded_ladder_floor(spec)
    _remove_embedded_ladder_lights(spec)
    _add_ladder_to_g1_spec(spec)
    return spec


def _remove_embedded_ladder_floor(spec: mujoco.MjSpec) -> None:
    """Keep the scene-owned terrain as the only ladder ground plane.

    The canonical G1 XML contains a world ``floor`` plane, while the ladder
    ``SceneCfg`` also creates its own plane terrain.  Leaving both at z=0
    produces duplicate contacts and visible z-fighting in rendered playback.
    """

    for geom in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM):
        if geom.name != "floor":
            continue
        if geom.type != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError(
                "The canonical G1 geom named 'floor' must be a plane so the "
                "ladder scene can replace it with SceneCfg terrain"
            )
        spec.delete(geom)
        return


def _remove_embedded_ladder_lights(spec: mujoco.MjSpec) -> None:
    """Use only the scene-owned light for stable ladder video rendering."""

    for light in list(spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_LIGHT)):
        spec.delete(light)


def make_g1_training_robot_cfg(robot_xml: str | Path | None = None):
    robot_cfg = get_g1_robot_cfg()
    robot_cfg.articulation = deepcopy(robot_cfg.articulation)
    xml_path = resolve_g1_training_xml(robot_xml)
    robot_cfg.spec_fn = partial(_get_g1_training_spec, xml_path)
    return robot_cfg


def make_g1_ladder_training_robot_cfg(robot_xml: str | Path | None = None):
    """Build the ladder robot from the canonical G1 XML plus generated props."""

    robot_cfg = get_g1_robot_cfg()
    robot_cfg.articulation = deepcopy(robot_cfg.articulation)
    arm_actuator_groups = 0
    for actuator_cfg in robot_cfg.articulation.actuators:
        target_names = tuple(actuator_cfg.target_names_expr)
        if target_names and all(
            any(part in target for part in ("shoulder", "elbow", "wrist"))
            for target in target_names
        ):
            actuator_cfg.effort_limit *= _LADDER_ARM_EFFORT_SCALE
            arm_actuator_groups += 1
    if arm_actuator_groups != 2:
        raise ValueError(
            "Expected two arm-only G1 actuator groups (shoulder/elbow/wrist), "
            f"found {arm_actuator_groups}; update the ladder effort-limit mapping"
        )
    robot_cfg.init_state = deepcopy(robot_cfg.init_state)
    robot_cfg.init_state.pos = _LADDER_INITIAL_ROOT_POS
    robot_cfg.init_state.joint_pos = dict(_LADDER_INITIAL_JOINT_POS)
    robot_cfg.init_state.joint_vel = {".*": 0.0}
    # G1's default collision editor disables every geom whose name does not
    # match ``.*_collision``.  The generated ladder uses semantic names, so it
    # must be explicitly re-enabled after the default editor runs.  A separate
    # collision bit lets the invisible face blocker stop only the trunk while
    # hands and feet can still reach and stand on the physical rungs.
    robot_cfg.collisions = (
        *robot_cfg.collisions,
        spec_cfg.CollisionCfg(
            geom_names_expr=(r"^(left|right)_ladder_rail_[0-9]{2}$",),
            contype=1,
            conaffinity=1,
            condim=4,
            friction=(1.4, 0.02, 0.002),
            solref=(0.005, 1.0),
            solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
            margin=0.002,
            disable_other_geoms=False,
        ),
        spec_cfg.CollisionCfg(
            geom_names_expr=(r"^(left|right)_ladder_rung_[0-9]{2}$",),
            contype=1,
            conaffinity=1,
            condim=4,
            friction=(1.8, 0.02, 0.002),
            solref=(0.005, 1.0),
            solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
            margin=0.002,
            disable_other_geoms=False,
        ),
        spec_cfg.CollisionCfg(
            geom_names_expr=(r"^(left|right)_ladder_body_blocker$",),
            contype=_LADDER_BODY_BLOCKER_MASK,
            conaffinity=_LADDER_BODY_BLOCKER_MASK,
            condim=1,
            solref=(0.005, 1.0),
            solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
            margin=0.002,
            disable_other_geoms=False,
        ),
        spec_cfg.CollisionCfg(
            geom_names_expr=(r"^(pelvis|torso|head)_collision$",),
            contype=1 | _LADDER_BODY_BLOCKER_MASK,
            conaffinity=1 | _LADDER_BODY_BLOCKER_MASK,
            condim=1,
            disable_other_geoms=False,
        ),
    )
    xml_path = resolve_g1_training_xml(robot_xml)
    robot_cfg.spec_fn = partial(_get_g1_ladder_training_spec, xml_path)
    return robot_cfg


def _apply_play_mode_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    for event_name in _TRAIN_ONLY_EVENTS:
        cfg.events.pop(event_name, None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"


def _add_history_obs_groups(
    cfg: ManagerBasedRlEnvCfg, history_length: int = 10
) -> None:
    cfg.observations["actor_history"] = ObservationGroupCfg(
        terms=deepcopy(cfg.observations["actor"].terms),
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
        history_length=history_length,
        flatten_history_dim=False,
    )
    cfg.observations["critic_history"] = ObservationGroupCfg(
        terms=deepcopy(cfg.observations["critic"].terms),
        concatenate_terms=True,
        enable_corruption=False,
        history_length=history_length,
        flatten_history_dim=False,
    )


_VELCMD_ACTOR_TERMS: dict[str, ObservationTermCfg] = {
    "robot_projected_gravity_b": ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "ref_anchor_lin_vel_b": ObservationTermCfg(
        func=mdp.ref_anchor_lin_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_anchor_ang_vel_b": ObservationTermCfg(
        func=mdp.ref_anchor_ang_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_projected_gravity_b": ObservationTermCfg(
        func=mdp.ref_projected_gravity_b,
        params={"command_name": "motion"},
    ),
    "ref_anchor_height": ObservationTermCfg(
        func=mdp.ref_anchor_height,
        params={"command_name": "motion"},
    ),
}

_VELCMD_CRITIC_TERMS: dict[str, ObservationTermCfg] = {
    "robot_projected_gravity_b": ObservationTermCfg(func=mdp.projected_gravity),
    "ref_anchor_lin_vel_b": ObservationTermCfg(
        func=mdp.ref_anchor_lin_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_anchor_ang_vel_b": ObservationTermCfg(
        func=mdp.ref_anchor_ang_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_projected_gravity_b": ObservationTermCfg(
        func=mdp.ref_projected_gravity_b,
        params={"command_name": "motion"},
    ),
    "ref_anchor_height": ObservationTermCfg(
        func=mdp.ref_anchor_height,
        params={"command_name": "motion"},
    ),
}


def _configure_self_collision_reward(cfg: ManagerBasedRlEnvCfg) -> None:
    excluded_body_names = (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    cfg.scene.sensors = (
        *tuple(getattr(cfg.scene, "sensors", ()) or ()),
        ContactSensorCfg(
            name="self_collision",
            # Exclude only primary wrist bodies; wrist vs torso is still caught by torso.
            primary=ContactMatch(
                mode="body",
                pattern=r".*",
                entity="robot",
                exclude=excluded_body_names,
            ),
            secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=4,
        ),
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={
            "sensor_name": "self_collision",
            "force_threshold": 1.0,
        },
    )


def _configure_feet_acc_reward(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.rewards["feet_acc"] = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=r".*ankle.*"),
        },
    )


def _configure_additional_wrist_pos_reward(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.rewards["additional_wrist_pos"] = RewardTermCfg(
        func=mdp.motion_relative_body_point_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.12,
            "body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
            "body_offsets": ((0.18, -0.025, 0.0), (0.18, 0.025, 0.0)),
        },
    )


def make_general_tracking_env_cfg(
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the General-Tracking-G1 training env."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": make_g1_training_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "torso_link"
    motion_cmd.body_names = _TRACKING_BODY_NAMES
    motion_cmd.motion_file = DEFAULT_TRAIN_MOTION_FILE
    motion_cmd.sampling_mode = "rewind"
    motion_cmd.window_steps = (0,)

    cfg.events["physics_material"].params["asset_cfg"].geom_names = r".*_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.events["randomize_rigid_body_mass"].params["asset_cfg"].body_names = (
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    _configure_self_collision_reward(cfg)
    _configure_feet_acc_reward(cfg)
    _configure_additional_wrist_pos_reward(cfg)
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    cfg.terminations["anchor_pos"].params["threshold"] = 0.25
    cfg.terminations["anchor_ori"].params["threshold"] = 1.0
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.25
    cfg.viewer.body_name = "torso_link"
    cfg.episode_length_s = 10.0
    if cfg.sim.njmax < 500:
        cfg.sim.njmax = 500

    actor_terms = {
        key: value
        for key, value in cfg.observations["actor"].terms.items()
        if key not in {"ref_anchor_pos_b", "robot_base_lin_vel_b"}
    }
    cfg.observations["actor"] = ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
    )

    cfg.observations["actor"].terms.update(deepcopy(_VELCMD_ACTOR_TERMS))
    cfg.observations["critic"].terms.update(deepcopy(_VELCMD_CRITIC_TERMS))

    _add_history_obs_groups(cfg)

    if play:
        _apply_play_mode_overrides(cfg)
        cfg.observations["actor_history"].enable_corruption = False
        cfg.observations["critic_history"].enable_corruption = False

    return cfg


def make_g1_ladder_rl_env_cfg(
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the G1 ladder task without motion imitation or reference data."""

    actor_terms = {
        "ladder_command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "ladder"},
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.0, n_max=1.0),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    critic_terms = {
        "ladder_command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "ladder"},
        ),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
        ),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    ladder_geometry_term = ObservationTermCfg(
        func=mdp.ladder_rung_tokens_torso,
        params={"command_name": "ladder"},
    )
    observations = {
        "actor": ObservationGroupCfg(
            actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "actor_ladder": ObservationGroupCfg(
            {"rung_tokens_torso": ladder_geometry_term},
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic_ladder": ObservationGroupCfg(
            {"rung_tokens_torso": deepcopy(ladder_geometry_term)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic_privileged": ObservationGroupCfg(
            {
                "ladder_privileged": ObservationTermCfg(
                    func=mdp.ladder_critic_privileged,
                    params={"command_name": "ladder"},
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
        )
    }

    commands = {
        "ladder": mdp.LadderClimbCommandCfg(
            entity_name="robot",
            hand_site_names=("left_grip_site", "right_grip_site"),
            foot_site_names=("left_foot", "right_foot"),
            foot_contact_sensor_name=_LADDER_FOOT_CONTACT_SENSOR,
            anchor_body_names=("left_grip_anchor_body", "right_grip_anchor_body"),
            weld_names=("left_grip_weld", "right_grip_weld"),
            rung_site_names=tuple(
                f"left_ladder_rung_{i:02d}_grip"
                for i in range(1, _LADDER_NUM_RUNGS + 1)
            ),
            torso_body_name="torso_link",
            pelvis_body_name="pelvis",
            first_moving_hand="left",
            first_moving_foot="left",
            start_rung=_LADDER_START_RUNG,
            initialize_on_reset=True,
            initial_foot_rung=_LADDER_INITIAL_FOOT_RUNG,
            curriculum_enabled=not play,
            curriculum_success_threshold=0.80,
            curriculum_window_size=100,
            curriculum_min_phase_steps=_LADDER_CURRICULUM_MIN_PHASE_STEPS,
            boundary_state_reset_prob=0.0 if play else 0.50,
            boundary_state_bank_size=0 if play else 1_024,
            stabilization_dwell_steps=50,
            stabilization_dwell_max_steps=100,
            hand_target_dwell_steps=3,
            foot_target_dwell_steps=5,
            max_stabilization_torso_speed=0.20,
            max_stabilization_joint_speed=1.0,
            max_stabilization_body_angular_speed=0.40,
            max_stabilization_waist_joint_speed=0.60,
            max_stabilization_support_offset_error=0.18,
            max_phase_torso_orientation_error=0.30,
            max_phase_support_offset_error=0.15,
            max_phase_completion_torso_speed=0.20,
            max_phase_completion_joint_speed=1.0,
            first_foot_max_body_drop=0.03,
            cycle_min_body_ascent=0.12,
            release_preload_dwell_steps=8,
            release_ramp_steps=20,
            release_final_dwell_steps=5,
            release_recovery_steps=2,
            pre_release_timeout_steps=300,
            max_release_torso_speed=0.12,
            max_release_torso_orientation_error=0.25,
            max_release_support_offset_error=0.12,
            release_soft_timeconst=0.18,
            release_soft_impedance=0.05,
            attach_distance=0.10,
            max_attach_speed=0.35,
            grip_half_span=0.18,
            foot_reach_distance=0.10,
            foot_support_distance=0.11,
            max_foot_speed=0.35,
            foot_target_height=_LADDER_RUNG_HALF_HEIGHT + 0.01,
            hand_foot_lead_rungs=3,
        ),
    }

    events = {
        "prepare_ladder_weld_model": EventTermCfg(
            func=mdp.prepare_ladder_weld_model,
            mode="startup",
        ),
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "physics_material": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    geom_names=r".*_collision$",
                ),
                "operation": "abs",
                "ranges": (0.5, 1.2),
                "shared_random": True,
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {
                    0: (-0.015, 0.015),
                    1: (-0.015, 0.015),
                    2: (-0.02, 0.02),
                },
            },
        ),
    }

    rewards = {
        "ladder_upward_progress": RewardTermCfg(
            func=mdp.LadderUpwardProgressReward,
            weight=20.0,
            params={"command_name": "ladder"},
        ),
        "ladder_foot_placement": RewardTermCfg(
            func=mdp.LadderFootPlacementReward,
            weight=8.0,
            params={
                "command_name": "ladder",
                "distance_std": 0.06,
                "max_abs_rate": 50.0,
            },
        ),
        "ladder_phase_progress": RewardTermCfg(
            func=mdp.LadderPhaseProgressReward,
            weight=8.0,
            params={
                "command_name": "ladder",
                "rung_spacing": _LADDER_HEIGHT / (_LADDER_NUM_RUNGS + 1),
                "reach_distance": 0.35,
                "first_hand_body_weight": 0.0,
                "second_hand_body_weight": 0.0,
                "foot_body_weight": 0.0,
                "release_progress_weight": 1.0,
                "unsupported_progress_scale": 0.25,
                "max_abs_rate": 10.0,
            },
        ),
        "ladder_phase_completed": RewardTermCfg(
            func=mdp.ladder_phase_completed,
            weight=25.0,
            params={"command_name": "ladder"},
        ),
        "ladder_stabilization_orientation": RewardTermCfg(
            func=mdp.ladder_stabilization_orientation_error_l2,
            weight=-1.0,
            params={"command_name": "ladder"},
        ),
        "ladder_failure": RewardTermCfg(
            func=mdp.ladder_failure_penalty,
            weight=-50.0,
            params={"command_name": "ladder"},
        ),
        "ladder_finished": RewardTermCfg(
            func=mdp.ladder_finished,
            weight=100.0,
            params={"command_name": "ladder"},
        ),
        "survival": RewardTermCfg(func=mdp.survival, weight=3.0),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.5),
        "joint_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
    }

    # Keep event bonuses and the episode height record shared. Each phase owns
    # its potential terms, with independent weights and separate reward logs.
    progress = rewards.pop("ladder_phase_progress")
    placement = rewards.pop("ladder_foot_placement")
    for phase in mdp.LadderPhase:
        for name, template in (("progress", progress), ("foot_placement", placement)):
            rewards[f"ladder_{phase.name.lower()}_{name}"] = RewardTermCfg(
                func=template.func,
                weight=template.weight,
                params={**template.params, "phase": int(phase)},
            )

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "success": TerminationTermCfg(
            func=mdp.ladder_success,
            params={"command_name": "ladder"},
        ),
        "curriculum_stage_complete": TerminationTermCfg(
            func=mdp.ladder_curriculum_stage_complete,
            params={"command_name": "ladder"},
            time_out=True,
        ),
        "pre_release_stalled": TerminationTermCfg(
            func=mdp.ladder_pre_release_stalled,
            params={"command_name": "ladder"},
        ),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.48},
        ),
        "root_too_low": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.30},
        ),
    }

    # Each MJWarp environment is an independent world. Keeping every origin at
    # zero aligns the fixed ladder with every batched robot instance.
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="plane",
                env_spacing=0.0,
                textures=(),
                materials=(_LADDER_FLOOR_MATERIAL,),
                lights=(_LADDER_SCENE_LIGHT,),
            ),
            entities={"robot": make_g1_ladder_training_robot_cfg()},
            sensors=(
                ContactSensorCfg(
                    name=_LADDER_FOOT_CONTACT_SENSOR,
                    primary=ContactMatch(
                        mode="body",
                        pattern=(
                            "left_ankle_roll_link",
                            "right_ankle_roll_link",
                        ),
                        entity="robot",
                    ),
                    secondary=ContactMatch(
                        mode="subtree",
                        pattern="left_ladder_body",
                        entity="robot",
                    ),
                    fields=("found", "force"),
                    reduce="maxforce",
                    num_slots=1,
                    history_length=4,
                ),
            ),
            num_envs=1,
            env_spacing=0.0,
            extent=4.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=4.0,
            elevation=-10.0,
            azimuth=120.0,
            enable_reflections=False,
            enable_shadows=False,
        ),
        sim=SimulationCfg(
            nconmax=200,
            njmax=1200,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=100,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )

    _configure_self_collision_reward(cfg)
    _configure_feet_acc_reward(cfg)
    _add_history_obs_groups(cfg)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.observations["actor_history"].enable_corruption = False
        cfg.events.pop("physics_material", None)
        cfg.events.pop("base_com", None)

    return cfg
