import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='RiverCrossing-v0',
    entry_point='gym_RiverCrossing.envs:RiverCrossingEnv',
    max_episode_steps=1000,
    reward_threshold=1.0,
    nondeterministic = False,
)