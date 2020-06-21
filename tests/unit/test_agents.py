"""Tests that the agent module works correctly"""
from unittest import TestCase
from unittest.mock import Mock

import gym
import numpy as np
import torch

from algos.common.agents import Agent, PolicyAgent, ValueAgent, default_states_preprocessor


class TestAgents(TestCase):

    def setUp(self) -> None:
        self.env = gym.make("CartPole-v0")
        self.state = [self.env.reset()]
        self.net = Mock()

    def test_base_agent(self):
        agent = Agent(self.net)
        action = agent(self.state, 'cuda:0')
        self.assertIsInstance(action, int)

    def test_preprocess(self):
        """Tests that given a list of numpy arrays can be converted to torch tensors"""
        state = np.random.rand(4)
        states = [state, state, state]
        torch_states = default_states_preprocessor(states)
        self.assertIsInstance(torch_states, torch.Tensor)
        self.assertEqual(torch_states.shape, (3, 4))


class TestValueAgent(TestCase):

    def setUp(self) -> None:
        self.env = gym.make("CartPole-v0")
        self.net = Mock(return_value=torch.Tensor([0.0, 100.0]))
        self.state = self.env.reset()
        self.states = [self.state, self.state, self.state]
        self.device = torch.device('cpu')
        self.value_agent = ValueAgent(self.net, self.env.action_space.n)

    def test_value_agent(self):

        actions = self.value_agent(self.states, self.device)
        self.assertIsInstance(actions, list)

    def test_value_agent_GET_ACTION_SINGLE(self):
        actions = self.value_agent.get_action([self.state], self.device)
        self.assertIsInstance(actions, list)
        self.assertIsInstance(actions[0], int)
        self.assertEqual(actions, [1])

    def test_value_agent_GET_ACTION_MULTI(self):
        actions = self.value_agent.get_action(self.states, self.device)
        self.assertIsInstance(actions, list)
        self.assertIsInstance(actions[0], int)
        self.assertEqual(actions, [1, 1, 1])

    def test_value_agent_RANDOM(self):
        actions = self.value_agent.get_random_action(self.states)
        self.assertIsInstance(actions, list)
        self.assertIsInstance(actions[0], int)


class TestPolicyAgent(TestCase):

    def setUp(self) -> None:
        self.env = gym.make("CartPole-v0")
        self.net = Mock(return_value=torch.Tensor([0.0, 100.0]))
        self.state = torch.tensor(self.env.reset())
        self.device = self.state.device

    def test_policy_agent(self):
        policy_agent = PolicyAgent(self.net)
        action = policy_agent(self.state, self.device)
        self.assertIsInstance(action, int)
        self.assertEqual(action, 1)

