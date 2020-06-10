"""Experience sources to be used as datasets for Ligthning DataLoaders"""
from collections import deque
from typing import List, Union, Tuple

import numpy as np
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


class NStepExperienceSource(ExperienceSource):

    def __init__(self, env: Env, agent: Agent, device, n_steps: int = 1):
        super().__init__(env, agent, device)
        self.n_steps = n_steps
        self.n_step_buffer = deque(maxlen=n_steps)

    def step(self) -> Experience:
        """
        Takes an n-step in the environment

        Returns:
            Experience
        """
        self.single_step()

        while len(self.n_step_buffer) < self.n_steps:
            self.single_step()

        reward, next_state, done = self.get_transition_info()
        first_experience = self.n_step_buffer[0]
        multi_step_experience = Experience(first_experience.state,
                                           first_experience.action,
                                           reward,
                                           done,
                                           next_state)

        return multi_step_experience

    def single_step(self) -> Experience:
        """
        Takes a  single step in the environment and appends it to the n-step buffer

        Returns:
            Experience
        """
        exp = super().step()
        self.n_step_buffer.append(exp)
        return exp

    def get_transition_info(self, gamma=0.9) -> Tuple[np.float, np.array, np.int]:
        """
        get the accumulated transition info for the n_step_buffer
        Args:
            gamma: discount factor

        Returns:
            multi step reward, final observation and done
        """
        last_experience = self.n_step_buffer[-1]
        final_state = last_experience.new_state
        done = last_experience.done
        reward = last_experience.reward

        # calculate reward
        # in reverse order, go through all the experiences up till the first experience
        for experience in reversed(list(self.n_step_buffer)[:-1]):
            reward_t = experience.reward
            new_state_t = experience.new_state
            done_t = experience.done

            reward = reward_t + gamma * reward * (1 - done_t)
            final_state, done = (new_state_t, done_t) if done_t else (final_state, done)

        return reward, final_state, done


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
