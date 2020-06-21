"""
Deep Reinforcement Learning: Deep Q-network (DQN)
This example is based on https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py
The template illustrates using Lightning for Reinforcement Learning. The example builds a basic DQN using the
classic CartPole environment.
To run the template just run:
python reinforce_learn_Qnet.py
After ~1500 steps, you will see the total_reward hitting the max score of 200. Open up TensorBoard to
see the metrics:
tensorboard --logdir default
"""
import argparse

import torch

from algos.common import wrappers
from algos.common.agents import ValueAgent
from algos.common.experience import NStepExperienceSource
from algos.common.memory import ReplayBuffer
from algos.dqn.model import DQNLightning

class NStepDQNLightning(DQNLightning):
    """ NStep DQN Model """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__(hparams)
        self.hparams = hparams

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.source = NStepExperienceSource(self.env, self.agent, device, n_steps=self.hparams.n_steps)
