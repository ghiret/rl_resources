import numpy as np
import torch


class GreedyStrategy:
    def __init__(self):
        """
        Initialize the Greedy strategy.

        This strategy always selects the action with the highest Q-value.
        """
        self.exploratory_action_taken = False

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)
