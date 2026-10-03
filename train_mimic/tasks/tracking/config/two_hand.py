"""Two sequential transfers, with optional measured-start sampling for exploration."""
from mjlab.managers import RewardTermCfg
from .first_hand import make_first_hand_env_cfg, make_first_hand_runner_cfg
from .env import _add_history_obs_groups
from ..mdp.two_hand import TwoHandCommandCfg, active_position_cost
from ..mdp.first_hand import orientation_tracking_cost
from ..mdp.ladder import LadderPhase


def make_two_hand_env_cfg(*, play=False):
    cfg = make_first_hand_env_cfg(play=play)
    settings = dict(vars(cfg.commands['ladder']))
    settings['fixed_max_unlocked_phase'] = int(LadderPhase.SECOND_HAND)
    cfg.commands['ladder'] = TwoHandCommandCfg(**settings, second_hand_reset_probability=0. if play else .5)
    cfg.episode_length_s = 40.
    cfg.rewards['reference_hand'] = RewardTermCfg(func=active_position_cost,weight=-3.,params={'std':.04})
    cfg.rewards['reference_supports'] = RewardTermCfg(func=active_position_cost,weight=-1.,params={'supports':True,'std':.03})
    cfg.rewards['reference_orientation'] = RewardTermCfg(func=orientation_tracking_cost,weight=-.5,params={'track_ids':(0,1,4,5)})
    _add_history_obs_groups(cfg)
    return cfg


def make_two_hand_runner_cfg():
    cfg = make_first_hand_runner_cfg()
    cfg.experiment_name = 'g1_ladder_two_hand_motion'
    cfg.max_iterations = 10000
    return cfg
