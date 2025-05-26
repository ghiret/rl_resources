import numpy as np
import torch


class EGreedyStrategy:
    def __init__(self, epsilon=0.1, seed=123):
        """
        Initialize the Epsilon-Greedy strategy.

        :param epsilon: Probability of taking a random action (exploration).
                        Default is 0.1 (10% exploration).
        :param seed: Random seed for reproducibility. Default is 123.
        """
        self.epsilon = epsilon
        self.exploratory_action_taken = None
        self._rng = np.random.RandomState(seed)

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()

        if self._rng.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = self._rng.randint(len(q_values))

        self.exploratory_action_taken = action != np.argmax(q_values)
        return action
