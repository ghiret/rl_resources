"""Tests for preprocessing utilities."""

import gymnasium as gym
import numpy as np
import pytest

from src.utils.preprocessing import normalize_observation, preprocess_env


def test_normalize_observation() -> None:
    """Test the normalize_observation function with various inputs."""
    # Test with simple array
    obs = np.array([1, 2, 3, 4, 5])
    normalized = normalize_observation(obs)
    assert np.all(normalized >= 0) and np.all(normalized <= 1)
    assert np.isclose(normalized[0], 0)  # min value should be 0
    assert np.isclose(normalized[-1], 1)  # max value should be 1

    # Test with zeros
    obs = np.zeros(5)
    normalized = normalize_observation(obs)
    assert np.all(normalized == 0)

    # Test with negative values
    obs = np.array([-2, -1, 0, 1, 2])
    normalized = normalize_observation(obs)
    assert np.all(normalized >= 0) and np.all(normalized <= 1)


@pytest.mark.skip(reason="Requires actual gym environment")
def test_preprocess_env() -> None:
    """Test the preprocess_env function with a gym environment."""
    env = gym.make("CartPole-v1")
    obs, info = preprocess_env(env)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert np.all(obs >= 0) and np.all(obs <= 1)
