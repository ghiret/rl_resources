import gymnasium as gym
import numpy as np

from .discounted_cartpole import DiscountedCartPole


class DummyEnv(gym.Env):
    def __init__(self):
        """
        Simulate the CartPole environment for testing purposes.

        It has a fixed observation space and action space, and simulates the behavior of
        the CartPole environment without actually implementing the physics.
        """
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self.x_threshold = 2.4
        self.theta_threshold_radians = 0.2095
        self.reset_called = False
        self.step_called = False
        self.next_obs = np.zeros(4)
        self.done = False

    @property
    def unwrapped(self):
        return self

    def reset(self, **_kwargs: dict) -> tuple[np.ndarray, dict]:
        """
        Reset the environment and return the initial observation and info.

        :param kwargs: Additional arguments for reset.
        :return: A tuple of (observation, info).
        """
        self.reset_called = True
        return self.next_obs, {}

    def step(self, _action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Simulate a step in the environment.

        This method simulates the behavior of the CartPole environment.
        It does not implement the actual physics but simulates the step behavior.

        :param action: The action taken by the agent.
        :return: A tuple of (observation, reward, done, truncated, info).
        """
        self.step_called = True
        # Simulate a step that does not fall
        obs = np.zeros(4)
        reward = 1.0
        done = False
        truncated = False
        info = {}
        return obs, reward, done, truncated, info


def test_reset_passes_through():
    """Test that reset passes through to the underlying environment and returns correct observation and info."""
    env = DummyEnv()
    wrapped = DiscountedCartPole(env)
    obs, info = wrapped.reset()
    assert env.reset_called
    assert np.allclose(obs, np.zeros(4))
    assert isinstance(info, dict)


def test_step_no_fall():
    """Test that step returns reward 0 when the pole does not fall."""
    env = DummyEnv()
    wrapped = DiscountedCartPole(env)
    obs, reward, _done, _truncated, info = wrapped.step(0)
    assert env.step_called
    assert reward == 0  # No fall, so reward should be 0
    assert np.allclose(obs, np.zeros(4))
    assert isinstance(info, dict)


def test_step_fall_x():
    """
    Test that step returns reward -1 when the pole falls due to x exceeding threshold.

    This simulates a fall by setting the observation's x value above the threshold.
    """
    env = DummyEnv()
    env.next_obs = np.array([3.0, 0.0, 0.0, 0.0])  # x > x_threshold

    def step(_):
        return env.next_obs, 1.0, True, False, {}

    env.step = step
    wrapped = DiscountedCartPole(env)
    _obs, reward, done, _truncated, _info = wrapped.step(0)
    assert reward == -1  # Should penalize for falling
    assert done is True


def test_step_fall_theta():
    """
    Test that step returns reward -1 when the pole falls due to theta exceeding threshold.

    This simulates a fall by setting the observation's theta value above the threshold.
    """
    env = DummyEnv()
    env.next_obs = np.array([0.0, 0.0, 0.3, 0.0])  # theta > theta_threshold_radians

    def step(_):
        return env.next_obs, 1.0, True, False, {}

    env.step = step
    wrapped = DiscountedCartPole(env)
    _obs, reward, done, _truncated, _info = wrapped.step(0)
    assert reward == -1  # Should penalize for falling
    assert done is True
