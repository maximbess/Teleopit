"""Registry wiring for the G1 tracking and RL-only ladder tasks."""

from mjlab.tasks.registry import register_mjlab_task

from train_mimic.tasks.tracking.config.constants import (
    GENERAL_TRACKING_EXPERIMENT_NAME,
    GENERAL_TRACKING_TASK,
    LADDER_RL_TASK,
)
from train_mimic.tasks.tracking.config.env import (
    make_g1_ladder_rl_env_cfg,
    make_general_tracking_env_cfg,
)
from train_mimic.tasks.tracking.config.rl import (
    make_g1_ladder_ppo_runner_cfg,
    make_general_tracking_ppo_runner_cfg,
)
from train_mimic.tasks.tracking.rl import (
    LadderOnPolicyRunner,
    MotionTrackingOnPolicyRunner,
)


register_mjlab_task(
    task_id=GENERAL_TRACKING_TASK,
    env_cfg=make_general_tracking_env_cfg(),
    play_env_cfg=make_general_tracking_env_cfg(play=True),
    rl_cfg=make_general_tracking_ppo_runner_cfg(
        experiment_name=GENERAL_TRACKING_EXPERIMENT_NAME,
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
)


register_mjlab_task(
    task_id=LADDER_RL_TASK,
    env_cfg=make_g1_ladder_rl_env_cfg(),
    play_env_cfg=make_g1_ladder_rl_env_cfg(play=True),
    rl_cfg=make_g1_ladder_ppo_runner_cfg(),
    runner_cls=LadderOnPolicyRunner,
)
