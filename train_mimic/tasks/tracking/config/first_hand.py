"""The single-hand residual motion task, independent of full ladder curriculum."""
from copy import deepcopy

from mjlab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from train_mimic.tasks.tracking import mdp
from train_mimic.tasks.tracking.mdp import first_hand as motion
from .env import make_g1_ladder_rl_env_cfg, _add_history_obs_groups
from .rl import make_g1_ladder_ppo_runner_cfg


def make_first_hand_env_cfg(*, play=False):
    cfg = make_g1_ladder_rl_env_cfg(play=play)
    settings = dict(vars(cfg.commands["ladder"]))
    settings.update(curriculum_enabled=False, boundary_state_reset_prob=0., boundary_state_bank_size=0,
                    fixed_max_unlocked_phase=int(mdp.LadderPhase.FIRST_HAND),
                    freeze_at_max_unlocked_phase=False, stabilization_dwell_max_steps=50)
    cfg.commands["ladder"] = motion.FirstHandCommandCfg(**settings)
    cfg.actions["joint_pos"] = motion.ResidualReferenceActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=.2,
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms = {
            "motion_stage": ObservationTermCfg(func=motion.stage_observation),
            **cfg.observations[group].terms,
        }
        cfg.observations[group].terms["motion_reference"] = ObservationTermCfg(func=motion.reference_observation)
    _add_history_obs_groups(cfg)
    # Begin with the same deterministic scene used to author the reference.
    cfg.events.pop("physics_material", None)
    cfg.events.pop("base_com", None)
    retained = {k: deepcopy(cfg.rewards[k]) for k in ("self_collisions", "ladder_unwanted_contact", "joint_limits", "feet_acc")}
    cfg.rewards = {
        **retained,
        "reference_hand": RewardTermCfg(func=motion.position_tracking_cost, weight=-3., params={"track_ids": (0,), "std": .04}),
        "reference_supports": RewardTermCfg(func=motion.position_tracking_cost, weight=-1., params={"track_ids": (1, 2, 3), "std": .03}),
        "reference_body": RewardTermCfg(func=motion.position_tracking_cost, weight=-1., params={"track_ids": (4, 5), "std": .05}),
        "reference_orientation": RewardTermCfg(func=motion.orientation_tracking_cost, weight=-.5),
        "reference_joints": RewardTermCfg(func=motion.joint_tracking_cost, weight=-.25),
        "reference_joint_velocity": RewardTermCfg(func=motion.joint_velocity_tracking_cost, weight=-.05),
        "torso_angular_velocity": RewardTermCfg(func=motion.body_angular_velocity_cost, weight=-.5,
                                                params={"body": "torso", "std": .4}),
        "pelvis_angular_velocity": RewardTermCfg(func=motion.body_angular_velocity_cost, weight=-.5,
                                                 params={"body": "pelvis", "std": .4}),
        "waist_velocity": RewardTermCfg(func=motion.waist_velocity_cost, weight=-.25, params={"std": .6}),
        "required_supports": RewardTermCfg(func=motion.support_reward, weight=.25),
        "lost_supports": RewardTermCfg(func=motion.missing_support_cost, weight=-4.),
        "new_hand_progress": RewardTermCfg(func=motion.event_reward, weight=10., params={"event": "progress"}),
        "new_grasp": RewardTermCfg(func=motion.event_reward, weight=10., params={"event": "grasp"}),
        "stable_success": RewardTermCfg(func=motion.event_reward, weight=100., params={"event": "success"}),
        "failure": RewardTermCfg(func=motion.failure_penalty, weight=-200.),
        "elapsed_time": RewardTermCfg(func=mdp.survival, weight=-.5),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-.02),
        "joint_limit_proximity": RewardTermCfg(func=motion.joint_limit_proximity, weight=-2., params={"margin_fraction": .15}),
    }
    cfg.terminations.pop("curriculum_stage_complete", None)
    cfg.terminations["motion_failed"] = TerminationTermCfg(func=motion.first_hand_failed)
    cfg.scene.num_envs = 64 if not play else 1
    cfg.episode_length_s = 20.
    cfg.scale_rewards_by_dt = True
    return cfg


def make_first_hand_runner_cfg():
    cfg = make_g1_ladder_ppo_runner_cfg()
    cfg.experiment_name = "g1_ladder_first_hand_motion"
    cfg.actor.class_name = "train_mimic.tasks.tracking.rl.first_hand_model:FirstHandTemporalCNNModel"
    cfg.critic.class_name = cfg.actor.class_name
    cfg.actor.hidden_dims = (512, 256, 128)
    cfg.critic.hidden_dims = (512, 256, 128)
    cfg.actor.distribution_cfg = {
        "class_name": "GaussianDistribution", "init_std": .1,
        "std_range": (.03, .2), "std_type": "scalar",
    }
    cfg.algorithm.entropy_coef = .0005
    cfg.max_iterations = 5000
    cfg.save_interval = 100
    return cfg
