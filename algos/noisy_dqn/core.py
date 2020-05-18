"""Core functions for the Noisy DQN"""
from typing import Tuple
import torch
from torch import nn
from algos.common.memory import Experience
from algos.dqn.core import Agent


class NoisyAgent(Agent):
    """
    Base Agent class handeling the interaction with the environment

    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def get_action(self, net: nn.Module, device: str) -> int:
        """
        Using the given network, decide what action to carry out
        using a noisy network

        Args:
            net: DQN network
            device: current device

        Returns:
            action
        """

        state = torch.tensor([self.state])

        if device not in ['cpu']:
            state = state.cuda(device)

        q_values = net(state)
        _, action = torch.max(q_values, dim=1)
        action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module,
                  epsilon: float = 0.0,
                  device: str = 'cpu',
                  render: bool = False) -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
            render: flag denoting whether to display the step

        Returns:
            reward, done
        """
        if render:
            self.env.render()

        if epsilon == 1.0:
            action = self.env.action_space.sample()
        else:
            action = self.get_action(net, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done
