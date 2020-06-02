import argparse
from unittest import TestCase
from unittest.mock import Mock

import gym
import torch
from torch.utils.data import DataLoader

from algos.common.agents import Agent
from algos.common.experience import OnPolicyExperienceStream
from algos.common.wrappers import ToTensor
from algos.dqn.model import DQNLightning
from algos.reinforce.model import ReinforceLightning


class TestReinforce(TestCase):

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))
        self.net = Mock()
        self.agent = Agent(self.net)
        self.xp_stream = OnPolicyExperienceStream(self.env, self.agent, episodes=4)
        self.rl_dataloader = DataLoader(self.xp_stream)

        parent_parser = argparse.ArgumentParser(add_help=False)
        parser = DQNLightning.add_model_specific_args(parent_parser)
        parser.add_argument("--algo", type=str, default="dqn", help="algorithm to use for training")
        args_list = [
            "--algo", "dqn",
            "--warm_start_steps", "500",
            "--episode_length", "100"
        ]
        self.hparams = parser.parse_args(args_list)

        self.model = ReinforceLightning(self.hparams)

    def test_loss(self):
        """Test the reinforce loss function"""

        for i_batch, batch in enumerate(self.rl_dataloader):
            exp_batch = batch

            batch_qvals, batch_states, batch_actions = self.model.process_batch(exp_batch)

            loss = self.model.loss(batch_qvals, batch_states, batch_actions)

            self.assertIsInstance(loss, torch.Tensor)
            break

    def test_get_qvals(self):
        """Test that given an batch of episodes that it will return a list of qvals for each episode"""
        batch_qvals = []
        for i_batch, batch in enumerate(self.rl_dataloader):

            for episode in batch:
                rewards = [step[2] for step in episode]
                batch_qvals.append(self.model.calc_qvals(rewards))

            self.assertEqual(len(batch_qvals), len(batch))
            self.assertIsInstance(batch_qvals[0][0], torch.Tensor)
            self.assertEqual(batch_qvals[0][0], (batch_qvals[0][1] * self.hparams.gamma) + 1.0)
            break

    def test_process_batch(self):
        """Test that given a batch of episodes that it will return the q_vals, the states and the actions"""

        for i_batch, batch in enumerate(self.rl_dataloader):

            q_vals, states, actions = self.model.process_batch(batch)

            self.assertEqual(len(q_vals), len(batch))
            self.assertEqual(len(states), len(batch))
            self.assertEqual(len(actions), len(batch))

