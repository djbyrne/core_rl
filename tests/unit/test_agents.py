"""Tests that the agent module works correctly"""
from unittest import TestCase
from unittest.mock import Mock

import gym
import torch

from algos.common.agents import Agent, PolicyAgent


class TestAgents(TestCase):

    def setUp(self) -> None:
        self.env = gym.make("CartPole-v0")
        self.state = self.env.reset()
        self.net = Mock()

    def test_base_agent(self):
        agent = Agent(self.net)
        action = agent(self.state, 'cuda:0')
        self.assertIsInstance(action, int)


class TestPolicyAgent(TestCase):

    def setUp(self) -> None:
        self.env = gym.make("CartPole-v0")
        self.net = Mock(return_value=torch.Tensor([0.0, 100.0]))
        self.state = torch.tensor(self.env.reset())
        self.device = 'cpu'

    def test_policy_agent(self):
        policy_agent = PolicyAgent(self.net)
        action = policy_agent(self.state, self.device)
        self.assertIsInstance(action, int)
        self.assertEqual(action, 1)

