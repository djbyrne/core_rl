from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import gym
import torch
from torch.utils.data import DataLoader

from algos.common.agents import Agent
from algos.common.experience import EpisodicExperienceStream, RLDataset, ExperienceSource
from algos.common.memory import Experience
from algos.common.wrappers import ToTensor


class DummyAgent(Agent):
    def __call__(self, states, agent_states):
        return 0


class TestEpisodicExperience(TestCase):
    """Test the standard experience stream"""

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))
        self.net = Mock()
        self.agent = Agent(self.net)
        self.xp_stream = EpisodicExperienceStream(self.env, self.agent, device=Mock(), episodes=4)
        self.rl_dataloader = DataLoader(self.xp_stream)

    def test_experience_stream_SINGLE_EPISODE(self):
        """Check that the experience stream gives 1 full episode per batch"""
        self.xp_stream.episodes = 1

        for i_batch, batch in enumerate(self.rl_dataloader):
            self.assertEqual(len(batch), 1)
            self.assertIsInstance(batch[0][0], Experience)
            self.assertEqual(batch[0][-1].done, True)

    def test_experience_stream_MULTI_EPISODE(self):
        """Check that the experience stream gives 4 full episodes per batch"""
        self.xp_stream.episodes = 4

        for i_batch, batch in enumerate(self.rl_dataloader):
            self.assertEqual(len(batch), 4)
            self.assertIsInstance(batch[0][0], Experience)
            self.assertEqual(batch[0][-1].done, True)


class TestExperienceSource(TestCase):

    def setUp(self) -> None:
        self.net = Mock()
        self.agent = DummyAgent(net=self.net)
        self.env = gym.make("CartPole-v0")
        self.source = ExperienceSource(self.env, self.agent, Mock())

    def test_step(self):
        exp = self.source.step()
        self.assertEqual(len(exp), 5)


class TestRLDataset(TestCase):

    def setUp(self) -> None:
        mock_states = np.random.rand(32, 4, 84, 84)
        mock_action = np.random.rand(32)
        mock_rewards = np.random.rand(32)
        mock_dones = np.random.rand(32)
        mock_next_states = np.random.rand(32, 4, 84, 84)
        self.sample = mock_states, mock_action, mock_rewards, mock_dones, mock_next_states

        self.buffer = Mock()
        self.buffer.sample = Mock(return_value=self.sample)
        self.dataset = RLDataset(buffer=self.buffer, sample_size=32)
        self.dl = DataLoader(self.dataset, batch_size=32)

    def test_rl_dataset_batch(self):
        """test that the dataset gives the correct batch"""

        for i_batch, sample_batched in enumerate(self.dl):
            self.assertIsInstance(sample_batched, list)
            self.assertEqual(sample_batched[0].shape, torch.Size([32, 4, 84, 84]))
            self.assertEqual(sample_batched[1].shape, torch.Size([32]))
            self.assertEqual(sample_batched[2].shape, torch.Size([32]))
            self.assertEqual(sample_batched[3].shape, torch.Size([32]))
            self.assertEqual(sample_batched[4].shape, torch.Size([32, 4, 84, 84]))



