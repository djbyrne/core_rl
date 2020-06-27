"""Module containing basic type of agents used by the various algorithms"""
from random import randint
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def default_states_preprocessor(states: List) -> List[torch.Tensor]:
    """
    Convert list of states into the form suitable for model. By default we assume Variable

    Args:
        list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor([np_states])


class Agent:
    """Basic agent that always returns 0"""

    def __init__(self, net: nn.Module):
        self.net = net

    @torch.no_grad()
    def __call__(self, states: List[torch.Tensor], device: str) -> List[int]:
        """
        Using the given network, decide what action to carry

        Args:
            states: current state of the environment
            device: device used for current batch
        Returns:
            action
        """
        assert isinstance(states, list)
        return 0


class ValueAgent(Agent):
    """Value based agent that returns an action based on the Q values from the network"""
    def __init__(self, net: nn.Module, action_space: int, eps_start: float = 1.0,
                 eps_end: float = 0.2, eps_frames: float = 1000):
        super().__init__(net)
        self.action_space = action_space
        self.eps_start = eps_start
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_frames = eps_frames

    @torch.no_grad()
    def __call__(self, state: np.ndarray, device: str) -> List[int]:
        """
        Takes in the current state and returns the action based on the agents policy

        Args:
            states: current state of the environment
            device: the device used for the current batch

        Returns:
            action defined by policy
        """
        # assert isinstance(states, list)

        if np.random.random() < self.epsilon:
            action = self.get_random_action()
        else:
            action = self.get_action(state, device)

        return action

    def get_random_action(self) -> List[int]:
        """returns a random action"""
        return randint(0, self.action_space - 1)

    def get_action(self, state: np.ndarray, device: torch.device) -> List[torch.Tensor]:
        """
            Returns the best action based on the Q values of the network
            Args:
                state: current state of the environment
                device: the device used for the current batch
            Returns:
                action defined by Q values
        """
        # assert len(state.shape) == 1
        torch_state = default_states_preprocessor(state)

        if device.type != 'cpu':
            torch_state = torch_state.cuda(device)

        q_values = self.net(torch_state)
        _, action = torch.max(q_values, dim=1)

        return action.item()

    def update_epsilon(self, step: int) -> None:
        """
        Updates the epsilon value based on the current step

        Args:
            step: current global step
        """
        self.epsilon = max(self.eps_end, self.eps_start - (step + 1) / self.eps_frames)


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
        if device.type != 'cpu':
            state = state.cuda(device)

        # get the logits and pass through softmax for probability distribution
        probabilities = F.softmax(self.net(state))
        prob_np = probabilities.data.cpu().numpy()

        # take the numpy values and randomly select action based on prob distribution
        action = np.random.choice(len(prob_np), p=prob_np)

        return action
