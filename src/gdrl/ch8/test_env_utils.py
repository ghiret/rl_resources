import gymnasium as gym
from gymnasium import wrappers

from .env_utils import get_make_env_fn


def test_make_env_fn_basic():
    """Test that make_env_fn creates a basic Gymnasium environment."""
    make_env_fn, _ = get_make_env_fn()
    env = make_env_fn("CartPole-v1")
    assert isinstance(env, gym.Env)
    obs, info = env.reset()
    assert isinstance(
        obs, (list, tuple, dict, float, int, type(env.observation_space.sample()))
    )
    env.close()


def test_make_env_fn_with_seed():
    """Test that make_env_fn sets the environment seed."""
    make_env_fn, _ = get_make_env_fn()
    env = make_env_fn("CartPole-v1", seed=123)
    assert hasattr(env.action_space, "seed")
    env.close()


def test_make_env_fn_with_monitor():
    """Test that make_env_fn wraps the environment with RecordEpisodeStatistics when monitor_mode is set."""
    make_env_fn, _ = get_make_env_fn()
    env = make_env_fn("CartPole-v1", monitor_mode="evaluation")
    # Should be wrapped with Monitor
    assert isinstance(env, wrappers.RecordEpisodeStatistics)
    env.close()


def test_make_env_fn_with_inner_outer_wrappers():
    """Test that make_env_fn applies inner and outer wrappers."""

    class DummyWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.dummy_wrapped = True

    make_env_fn, _ = get_make_env_fn()
    env = make_env_fn(
        "CartPole-v1",
        inner_wrappers=[DummyWrapper],
        outer_wrappers=[DummyWrapper],
    )
    # Should be wrapped twice
    assert hasattr(env, "dummy_wrapped")
    assert hasattr(env.env, "dummy_wrapped")
    env.close()
