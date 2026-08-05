"""Public constants for supported tracking tasks."""

DEFAULT_TRAIN_MOTION_FILE = "data/datasets_precomputed"
GENERAL_TRACKING_TASK = "General-Tracking-G1"
GENERAL_TRACKING_EXPERIMENT_NAME = "g1_general_tracking"

LADDER_RL_TASK = "G1-Ladder-Climb-RL"
LADDER_RL_EXPERIMENT_NAME = "g1_ladder_rl"

TRACKING_TASKS = (GENERAL_TRACKING_TASK,)
LADDER_TASKS = (LADDER_RL_TASK,)
SUPPORTED_TASKS = TRACKING_TASKS + LADDER_TASKS
