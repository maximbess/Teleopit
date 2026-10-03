"""Registry wiring for tracking, ladder RL and first-hand residual motion tasks."""

from mjlab.tasks.registry import register_mjlab_task

from train_mimic.tasks.tracking.config.constants import (
    GENERAL_TRACKING_EXPERIMENT_NAME,
    GENERAL_TRACKING_TASK,
    LADDER_RL_TASK,
    FIRST_HAND_MOTION_TASK,
    TWO_HAND_MOTION_TASK,
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
from .first_hand import make_first_hand_env_cfg, make_first_hand_runner_cfg
from train_mimic.tasks.tracking.rl.first_hand_runner import FirstHandOnPolicyRunner
from .two_hand import make_two_hand_env_cfg, make_two_hand_runner_cfg
from train_mimic.tasks.tracking.rl.two_hand_runner import TwoHandOnPolicyRunner


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

register_mjlab_task(
    task_id=FIRST_HAND_MOTION_TASK,
    env_cfg=make_first_hand_env_cfg(),
    play_env_cfg=make_first_hand_env_cfg(play=True),
    rl_cfg=make_first_hand_runner_cfg(),
    runner_cls=FirstHandOnPolicyRunner,
)

register_mjlab_task(
    task_id=TWO_HAND_MOTION_TASK,
    env_cfg=make_two_hand_env_cfg(),
    play_env_cfg=make_two_hand_env_cfg(play=True),
    rl_cfg=make_two_hand_runner_cfg(),
    runner_cls=TwoHandOnPolicyRunner,
)
