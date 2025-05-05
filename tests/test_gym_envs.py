import gymnasium as gym

from src import gym_walk


def test_gym_walk_import():
    """Test that the SlipperyWalkFive environment can be imported and created."""
    env = gym.make("SlipperyWalkFive-v0")
    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None

    # Test environment reset
    obs, info = env.reset()
    assert obs is not None
    assert info is not None

    # Test environment step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info is not None

    env.close()

    # Explicitly use gym_walk (even if just accessing the module itself)
    assert gym_walk is not None
