import numpy as np
import torch

from .greedy_strategy import GreedyStrategy


class DummyModel:
    def __call__(self, _state):
        # Return a fixed Q-value array for testing
        return torch.tensor([[1.0, 2.0, 3.0]])


def test_select_action_returns_argmax():
    """Test that select_action returns the action with the highest Q-value."""
    strategy = GreedyStrategy()
    model = DummyModel()
    state = np.zeros((3,))
    action = strategy.select_action(model, state)
    optimal_action = 2  # argmax([1,2,3]) == 2
    assert action == optimal_action


def test_exploratory_action_taken_is_false():
    """Test that exploratory_action_taken is False after select_action."""
    strategy = GreedyStrategy()
    model = DummyModel()
    state = np.zeros((3,))
    strategy.select_action(model, state)
    assert strategy.exploratory_action_taken is False
