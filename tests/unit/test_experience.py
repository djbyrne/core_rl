from unittest import TestCase
from unittest.mock import Mock

import gym
from torch.utils.data import DataLoader

from algos.common.agents import Agent
from algos.common.experience import OnPolicyExperienceStream
from algos.common.memory import Experience
from algos.common.wrappers import ToTensor


class TestExperienceSteam(TestCase):
    """Test the standard experience stream"""

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))
        self.net = Mock()
        self.agent = Agent(self.net)
        self.xp_stream = OnPolicyExperienceStream(self.env, self.agent, episodes=4)
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


    def test_experience_stream_DONE(self):
        mock_env = Mock()
        mock_env.step = Mock(return_value=(0, 0, True, 0))
        mock_env.reset = Mock(reutrn_value=0)
        self.xp_stream.env = mock_env

        for i_batch, batch in enumerate(self.rl_dataloader):
            self.assertIsInstance(batch[0][0], Experience)
            self.assertEqual(batch[0][-1].done, True)
            self.assertTrue(self.xp_stream.env.reset.call_count, self.xp_stream.episodes)
            break
