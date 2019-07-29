from gym.envs.registration import register

register(
    id='qrrt-v0',
    entry_point='gym_qrrt.envs:qrrtEnv',
)
