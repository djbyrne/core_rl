"""Tests that the agent module works correctly"""
from unittest import TestCase

import gym

from algos.common.agents import Agent


class TestAgents(TestCase):

    def setUp(self) -> None:
        self.env = gym.make("CartPole-v0")
        self.state = self.env.reset()

    def test_base_agent(self):
        agent = Agent()
        action = agent(self.state)
        self.assertIsInstance(action, int)


