from unittest import TestCase
from unittest.mock import Mock

import gym
import torch
from torch.utils.data import DataLoader

from algos.common.agents import Agent
from algos.common.experience import ExperienceStream
from algos.common.memory import Experience
from algos.common.wrappers import ToTensor


class TestExperienceSteam(TestCase):
    """Test the standard experience stream"""

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))
        self.net = Mock()
        self.agent = Agent(self.net)
        self.xp_stream = ExperienceStream(self.env, self.agent)
        self.rl_dataloader = DataLoader(self.xp_stream)

    def test_experience_stream(self):

        for i_batch, experience in enumerate(self.rl_dataloader):
            self.assertIsInstance(experience, Experience)
            self.assertTrue(torch.all(self.xp_stream.state.eq(experience.new_state)))

            if i_batch > 5:
                break

    def test_experience_stream_DONE(self):
        mock_env = Mock()
        mock_env.step = Mock(return_value=(0, 0, True, 0))
        mock_env.reset = Mock(reutrn_value=0)
        self.xp_stream.env = mock_env


        for i_batch, experience in enumerate(self.rl_dataloader):
            self.assertIsInstance(experience, Experience)
            self.assertTrue(self.xp_stream.env.reset.call_count, 1)
            break
