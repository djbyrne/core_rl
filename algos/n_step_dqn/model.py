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

from algos.common import wrappers
from algos.common.memory import MultiStepBuffer
from algos.common.networks import DuelingCNN
from algos.dqn.core import Agent
from algos.dqn.model import DQNLightning


class NStepDQNLightning(DQNLightning):
    """ NStep DQN Model """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__(hparams)
        self.hparams = hparams

        self.env = wrappers.make_env(self.hparams.env)
        self.env.seed(123)

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.net = None
        self.target_net = None
        self.build_networks()

        self.buffer = MultiStepBuffer(self.hparams.replay_size, n_step=4)
        self.agent = Agent(self.env, self.buffer)

        self.total_reward = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0

        self.populate(self.hparams.warm_start_steps)
