"""Experience sources to be used as datasets for Ligthning DataLoaders"""
from gym import Env
from torch.utils.data import IterableDataset
from typing import List

from algos.common.agents import Agent
from algos.common.memory import Experience


class OnPolicyExperienceStream(IterableDataset):
    """
    Basic experience stream that iteratively yield the current experience of the agent in the env

    Args:
        env: Environmen that is being used
        agent: Agent being used to make decisions
    """

    def __init__(self, env: Env, agent: Agent, episodes: int = 1):
        self.env = env
        self.agent = agent
        self.state = self.env.reset()
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
        action = self.agent(self.state)
        new_state, reward, done, _ = self.env.step(action)
        self.state = new_state

        experience = Experience(state=self.state, action=action, reward=reward, new_state=new_state, done=done)

        if done:
            self.state = self.env.reset()

        return experience
