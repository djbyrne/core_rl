"""Module containing basic type of agents used by the various algorithms"""
from numpy.core.multiarray import ndarray


class Agent:
    """Basic agent that performs and action with"""

    def __init__(self):
        pass

    def __call__(self, state: ndarray) -> int:
        """Give the current state the agent returns an action based on its policy"""
        return 0
