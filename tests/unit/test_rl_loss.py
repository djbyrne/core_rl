"""
Test RL Loss Functions
"""

from unittest import TestCase

import torch
import numpy as np

from losses.loss import dqn_loss, double_dqn_loss, per_dqn_loss
from algos.common.networks import CNN
from algos.common.wrappers import make_env


class TestRLLoss(TestCase):

    def setUp(self) -> None:

        self.state = torch.rand(32, 4, 84, 84)
        self.next_state = torch.rand(32, 4, 84, 84)
        self.action = torch.ones([32])
        self.reward = torch.ones([32])
        self.done = torch.zeros([32]).long()

        self.batch = (self.state, self.action, self.reward, self.done, self.next_state)

        self.env = make_env("PongNoFrameskip-v4")
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.net = CNN(self.obs_shape, self.n_actions)
        self.target_net = CNN(self.obs_shape, self.n_actions)

    def test_dqn_loss(self):
        """Test the dqn loss function"""

        loss = dqn_loss(self.batch, self.net, self.target_net)
        self.assertIsInstance(loss, torch.Tensor)

    def test_double_dqn_loss(self):
        """Test the double dqn loss function"""

        loss = double_dqn_loss(self.batch, self.net, self.target_net)
        self.assertIsInstance(loss, torch.Tensor)

    def test_per_dqn_loss(self):
        """Test the double dqn loss function"""
        prios = torch.ones([32])

        loss, batch_weights = per_dqn_loss(self.batch, prios, self.net, self.target_net)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(batch_weights, np.ndarray)
