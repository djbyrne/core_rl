"""Module containing basic type of agents used by the various algorithms"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Agent:
    """Basic agent that always returns 0"""

    def __init__(self, net: nn.Module):
        self.net = net

    def __call__(self, state: torch.Tensor) -> int:
        """
        Using the given network, decide what action to carry

        Args:
            state: current state of the environment

        Returns:
            action
        """
        return 0

class PolicyAgent(Agent):
    """Policy based agent that returns an action based on the networks policy"""

    def __call__(self, state: torch.Tensor, device: str) -> int:
        """
        Takes in the current state and returns the action based on the agents policy

        Args:
            state: current state of the environment
            device: the device used for the current batch

        Returns:
            action defined by policy
        """

        if device not in ['cpu']:
            state = state.cuda(device)

        # get the logits and pass through softmax for probability distribution
        probabilities = F.softmax(self.net(state))
        prob_np = probabilities.data.cpu().numpy()

        # take the numpy values and randomly select action based on prob distribution
        action = np.random.choice(len(prob_np), p=prob_np)

        return action
