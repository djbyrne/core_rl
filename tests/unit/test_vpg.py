import argparse
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import gym
import torch
from torch.utils.data import DataLoader

from algos.common import cli
from algos.common.agents import Agent
from algos.common.experience import EpisodicExperienceStream
from algos.common.networks import MLP
from algos.common.wrappers import ToTensor
from algos.vanilla_policy_gradient.model import VPGLightning


class TestVPG(TestCase):

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.net = MLP(self.obs_shape, self.n_actions)
        self.agent = Agent(self.net)
        self.xp_stream = EpisodicExperienceStream(self.env, self.agent, Mock(), episodes=4)
        self.rl_dataloader = DataLoader(self.xp_stream)

        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser = cli.add_base_args(parent=parent_parser)
        parent_parser = VPGLightning.add_model_specific_args(parent_parser)
        args_list = [
            "--episode_length", "100",
        ]
        self.hparams = parent_parser.parse_args(args_list)

        self.model = VPGLightning(self.hparams)

    def test_calc_q_vals(self):
        rewards = [torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1)]
        gt_qvals = np.array([ 1.4652743,   0.49497533, -0.4851246,  -1.4751246 ])

        qvals = self.model.calc_qvals(rewards)
        qvals = torch.stack(qvals).numpy()

        self.assertEqual(gt_qvals.all(), qvals.all())

    def test_loss(self):
        """Test the vpg loss function"""
        self.model.net = self.net
        self.model.agent = self.agent

        for i_batch, batch in enumerate(self.rl_dataloader):
            exp_batch = batch

            batch_qvals, batch_states, batch_actions, _ = self.model.process_batch(exp_batch)

            loss = self.model.loss(batch_qvals, batch_states, batch_actions)

            self.assertIsInstance(loss, torch.Tensor)
            break
