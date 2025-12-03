from envs.mp_env import MPEnv
# import sac_discrete_per
from gym.envs.registration import register

# register(
#     id='mp-v0',
#     entry_point='mprnn.mp_env.envs:MPEnv',
# )

register(
    id='mp-v0',
    entry_point='MP-RNN.envs:MPEnv',
)