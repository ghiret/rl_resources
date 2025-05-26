"""Code for chapter 8 of the GDRL book."""

from .discounted_cartpole import DiscountedCartPole
from .egreedy_strategy import EGreedyStrategy
from .fcq import FCQ
from .greedy_strategy import GreedyStrategy
from .nfq import BEEP, NFQ

__all__ = [
    "FCQ",
    "EGreedyStrategy",
    "GreedyStrategy",
    "DiscountedCartPole",
    "NFQ",
    "BEEP",
]
