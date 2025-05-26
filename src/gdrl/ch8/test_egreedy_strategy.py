import numpy as np
import torch

from .egreedy_strategy import EGreedyStrategy


class DummyModel:
    def __call__(self, _state):
        # Return a fixed Q-value array for testing
        return torch.tensor([[1.0, 2.0, 3.0]])


def test_select_action_returns_valid_action():
    """Test that select_action returns a valid action index."""
    strategy = EGreedyStrategy(epsilon=0.0, seed=42)  # Always greedy, deterministic
    model = DummyModel()
    state = np.zeros((3,))
    action = strategy.select_action(model, state)
    assert action in [0, 1, 2]


def test_select_action_greedy():
    """Test that select_action returns the greedy action when epsilon=0."""
    strategy = EGreedyStrategy(epsilon=0.0, seed=42)  # Always greedy, deterministic
    model = DummyModel()
    state = np.zeros((3,))
    action = strategy.select_action(model, state)
    optimal_action = 2  # argmax([1,2,3]) == 2
    assert action == optimal_action
    assert bool(strategy.exploratory_action_taken) is False


def test_select_action_exploratory():
    """Test that select_action returns an exploratory action when epsilon=1.0."""
    strategy = EGreedyStrategy(
        epsilon=1.0, seed=42
    )  # Always exploratory, deterministic
    model = DummyModel()
    state = np.zeros((3,))
    action = strategy.select_action(model, state)
    # With a fixed seed, the exploratory action is deterministic
    assert action == 0  # This value is determined by the seed
    assert bool(strategy.exploratory_action_taken) is True
