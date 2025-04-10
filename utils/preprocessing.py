"""Preprocessing utilities for reinforcement learning environments."""

from typing import Any

import gymnasium as gym
import numpy as np


def normalize_observation(obs: np.ndarray) -> np.ndarray:
    """
    Normalize observation values to range [0, 1].

    Args:
    ----
        obs: Input observation array

    Returns:
    -------
        Normalized observation array

    """
    return (obs - obs.min()) / (obs.max() - obs.min() + 1e-8)


def preprocess_env(env: gym.Env) -> tuple[Any, Any]:
    """
    Preprocess environment observation and return initial state.

    Args:
    ----
        env: Gymnasium environment

    Returns:
    -------
        Tuple of (observation, info)

    """
    obs, info = env.reset()
    obs = normalize_observation(obs)
    return obs, info
