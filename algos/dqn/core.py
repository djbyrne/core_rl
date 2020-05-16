"""Core functions for DQN Methods"""

from typing import Tuple
import numpy as np
import gym
import torch
from torch import nn
from torch.utils.data.dataset import IterableDataset

from algos.common.memory import ReplayBuffer, Experience


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 1) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for idx, _ in enumerate(dones):
            yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

    def __getitem__(self, item):
        """Not used"""
        return None


class Agent:
    """
    Base Agent class handeling the interaction with the environment

    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """ Resents the environment and updates the state"""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
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

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done
