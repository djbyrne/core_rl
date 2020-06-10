"""Experience sources to be used as datasets for Ligthning DataLoaders"""
from typing import List, Union, Tuple

import torch
from gym import Env
from torch.utils.data import IterableDataset
from algos.common.agents import Agent
from algos.common.memory import Experience, MeanBuffer, ReplayBuffer


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


class ExperienceSource(IterableDataset):
    """
    Basic single step experience source

    Args:
        env: Environment that is being used
        agent: Agent being used to make decisions
    """

    def __init__(self, env: Env, agent: Agent, device):
        self.env = env
        self.agent = agent
        self.state = self.env.reset()
        self.device = device

    def step(self) -> Experience:
        """Takes a single step through the environment"""
        action = self.agent(self.state, self.device)
        new_state, reward, done, _ = self.env.step(action)
        experience = Experience(state=self.state, action=action, reward=reward, new_state=new_state, done=done)
        self.state = new_state

        if done:
            self.state = self.env.reset()

        return experience


class EpisodicExperienceStream(ExperienceSource):
    """
    Basic experience stream that iteratively yield the current experience of the agent in the env

    Args:
        env: Environmen that is being used
        agent: Agent being used to make decisions
    """

    def __init__(self, env: Env, agent: Agent, device, episodes: int = 1):
        super().__init__(env, agent, device)
        self.episodes = episodes

    def __getitem__(self, item):
        return item

    def __iter__(self) -> List[Experience]:
        """
        Plays a step through the environment until the episode is complete

        Returns:
            Batch of all transitions for the entire episode
        """
        episode_steps, batch = [], []

        while len(batch) < self.episodes:
            exp = self.step()
            episode_steps.append(exp)

            if exp.done:
                batch.append(episode_steps)
                episode_steps = []

        yield batch

    def step(self) -> Experience:
        """Carries out a single step in the environment"""
        action = self.agent(self.state, self.device)
        new_state, reward, done, _ = self.env.step(action)
        experience = Experience(state=self.state, action=action, reward=reward, new_state=new_state, done=done)
        self.state = new_state

        if done:
            self.state = self.env.reset()

        return experience
