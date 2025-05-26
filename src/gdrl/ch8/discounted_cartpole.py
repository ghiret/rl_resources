from typing import Any

import gymnasium as gym


class DiscountedCartPole(gym.Wrapper):
    def __init__(self, env):
        """
        Modify the reward structure of the CartPole environment to penalize falling.

        This wrapper is designed to encourage the agent to keep the pole balanced
        for as long as possible by giving a negative reward when the pole falls.
        :param env: The CartPole environment to wrap.
        """
        gym.Wrapper.__init__(self, env)

    def reset(self, **kwargs: dict[str, Any]):
        return self.env.reset(**kwargs)

    def step(self, action: int):
        """
        Modify the reward structure of the CartPole environment to penalize falling.

        This wrapper is designed to encourage the agent to keep the pole balanced
        for as long as possible by giving a negative reward when the pole falls.
        :param action: The action taken by the agent.
        :return: A tuple of (observation, reward, done, truncated, info).
        """
        obs, _original_reward, terminated, truncated, info = self.env.step(action)
        (x, _x_dot, theta, _theta_dot) = obs
        pole_fell = (
            x < -self.env.unwrapped.x_threshold
            or x > self.env.unwrapped.x_threshold
            or theta < -self.env.unwrapped.theta_threshold_radians
            or theta > self.env.unwrapped.theta_threshold_radians
        )
        reward = -1 if pole_fell else 0
        return obs, reward, terminated, truncated, info
