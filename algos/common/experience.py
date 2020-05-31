import torch
from gym import Env
from torch.utils.data import IterableDataset

from algos.common.agents import Agent
from algos.common.memory import Experience


class ExperienceStream(IterableDataset):
    """
    Basic experience stream that iteratively yield the current experience of the agent in the env

    Args:
        env: Environmen that is being used
        agent: Agent being used to make decisions
    """

    def __init__(self, env: Env, agent: Agent):
        self.env = env
        self.agent = agent
        self.state = self.env.reset()

    def __iter__(self):
        action = self.agent(self.state)
        new_state, reward, done, _ = self.env.step(action)
        self.state = new_state

        experience = Experience(state=self.state, action=action, reward=reward, new_state=new_state, done=done)

        if done:
            self.state = self.env.reset()

        yield experience
