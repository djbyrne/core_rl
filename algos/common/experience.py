"""Experience sources to be used as datasets for Ligthning DataLoaders"""
from collections import deque
from typing import List, Tuple, Union

import numpy as np
import torch
from gym import Env
from torch.utils.data import IterableDataset
from algos.common.agents import Agent
from algos.common.memory import Experience, Buffer


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: Buffer, sample_size: int = 1) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for idx, _ in enumerate(dones):
            yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

    def __getitem__(self, item):
        """Not used"""
        return None


class PrioRLDataset(RLDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __iter__(self) -> Tuple:
        samples, indices, weights = self.buffer.sample(self.sample_size)

        states, actions, rewards, dones, new_states = samples
        for idx, _ in enumerate(dones):
            yield (states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]), indices[idx], weights[idx]


class ExperienceSource:
    """
    Basic single step experience source

    Args:
        env: Environment that is being used
        agent: Agent being used to make decisions
    """

    def __init__(self, env: Union[Env, List[Env]], agent: Agent, device: torch.device):
        assert isinstance(env, (Env, list))
        assert isinstance(agent, Agent)
        assert isinstance(device, torch.device)

        self.env_pool = env if isinstance(env, list) else [env]
        self.agent = agent
        self.states = []
        self.env_reward = []
        self.states = []
        self.action_list = []
        self._reset()
        self.device = device

    def _reset(self) -> None:
        """resets the env and state"""

        for env in self.env_pool:
            self.states.append(env.reset())

    def step(self) -> Tuple[List[Experience], List[float], List[bool]]:
        """Takes a single step through each environment in the pool"""
        agent_actions = self.agent(self.states, self.device)
        experiences, rewards, dones = [], [], []

        for idx, action in enumerate(agent_actions):
            new_state, reward, done, _ = self.env_pool[idx].step(action)
            exp = Experience(state=self.states[0], action=action, reward=reward, new_state=new_state, done=done)
            self.states[idx] = new_state
            experiences.append(exp)
            rewards.append(reward)
            dones.append(done)

            if done:
                self.states[idx] = self.env_pool[idx].reset()

        return experiences, rewards, dones

    def run_episode(self) -> List[float]:
        """
        Carries out a single episode and returns the total reward. This is used for testing

        Returns:
            List of total rewards for the number of environments run
        """
        done = False
        total_rewards = []

        while not done:
            _, reward, done = self.step()
            for idx, rew in enumerate(reward):
                if len(total_rewards) < len(reward):
                    total_rewards.append(rew)
                else:
                    total_rewards[idx] += reward[idx]

        return total_rewards


class NStepExperienceSource(ExperienceSource):
    """Expands upon the basic ExperienceSource by collecting experience across N steps"""
    def __init__(self, env_pool: List[Env], agent: Agent, device, n_steps: int = 1):
        super().__init__(env_pool, agent, device)
        self.n_steps = n_steps
        self.n_step_buffer = [deque(maxlen=n_steps) for _ in range(len(env_pool))]

    def step(self) -> Tuple[List[Experience], List[float], List[bool]]:
        """
        Takes an n-step in the environment

        Returns:
            Experience
        """
        self.single_step()
        mulit_experiences, rewards, dones = [], [], []

        for idx in range(len(self.env_pool)):
            buffer = self.n_step_buffer[idx]

            while len(buffer) < self.n_steps:
                self.single_step()

            reward, next_state, done = self.get_transition_info(buffer)
            first_experience = buffer[0]
            multi_step_experience = Experience(first_experience.state,
                                               first_experience.action,
                                               reward,
                                               done,
                                               next_state)

            mulit_experiences.append(multi_step_experience)
            rewards.append(reward)
            dones.append(done)

        return mulit_experiences, rewards, dones

    def single_step(self) -> List[Experience]:
        """
        Takes a  single step in the environment and appends it to the n-step buffer

        Returns:
            Experience
        """
        experiences, _, _ = super().step()
        for idx, exp in enumerate(experiences):
            self.n_step_buffer[idx].append(exp)
        return experiences

    def get_transition_info(self, buffer: deque, gamma=0.9) -> Tuple[np.float, np.array, np.int]:
        """
        get the accumulated transition info for the n_step_buffer
        Args:
            gamma: discount factor

        Returns:
            multi step reward, final observation and done
        """
        last_experience = buffer[-1]
        final_state = last_experience.new_state
        done = last_experience.done
        reward = last_experience.reward

        # calculate reward
        # in reverse order, go through all the experiences up till the first experience
        for experience in reversed(list(buffer)[:-1]):
            reward_t = experience.reward
            new_state_t = experience.new_state
            done_t = experience.done

            reward = reward_t + gamma * reward * (1 - done_t)
            final_state, done = (new_state_t, done_t) if done_t else (final_state, done)

        return reward, final_state, done


class EpisodicExperienceStream(ExperienceSource, IterableDataset):
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
        action = self.agent(self.states, self.device)
        new_state, reward, done, _ = self.env_pool[0].step(action)
        experience = Experience(state=self.states, action=action, reward=reward, new_state=new_state, done=done)
        self.state = new_state

        if done:
            self.state = self.env_pool[0].reset()

        return experience
