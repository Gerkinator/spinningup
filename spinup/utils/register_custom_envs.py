import gym


class register:
    def __init__(self):
        gym.envs.registration.register(id="FetchPush-jordan-v0",
                                        entry_point='RoboticFetchPush:FetchPush',
                                        max_episode_steps=200)