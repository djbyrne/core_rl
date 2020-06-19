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
from collections import OrderedDict
from typing import Tuple

import torch

from algos.common.networks import NoisyCNN
from algos.dqn.model import DQNLightning


class NoisyDQNLightning(DQNLightning):
    """ Noisy DQN Model """

    def build_networks(self) -> None:
        """Initializes the Noisy DQN train and target networks"""
        self.net = NoisyCNN(self.obs_shape, self.n_actions)
        self.target_net = NoisyCNN(self.obs_shape, self.n_actions)

    def on_train_start(self) -> None:
        """Set the agents epsilon to 0 as the exploration comes from the network"""
        self.agent.epsilon = 0.0

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """
        # step through environment with agent and add to buffer
        exp, reward, done = self.source.step()
        self.buffer.append(exp)

        self.episode_reward += reward
        self.episode_steps += 1

        # calculates training loss
        loss = self.loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        # Handle rewards and dones for each environment run
        for idx, rew in enumerate(reward):
            self.episode_reward[idx] += reward[idx]
            self.episode_steps[idx] += 1

            if done[idx]:
                self.total_reward = self.episode_reward[idx]
                self.reward_list.append(self.episode_reward[idx])
                self.avg_reward = sum(self.reward_list[-100:]) / 100
                self.episode_count += 1
                self.episode_reward[idx] = 0
                self.total_episode_steps = self.episode_steps[idx]
                self.episode_steps[idx] = 0

        # Soft update of target network
        if (self.global_step * len(self.source.env_pool)) % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {'total_reward': self.total_reward,
               'avg_reward': self.avg_reward,
               'train_loss': loss,
               'episode_steps': torch.tensor(self.total_episode_steps)
               }
        status = {'steps': torch.tensor(self.global_step).to(self.device),
                  'avg_reward': self.avg_reward,
                  'total_reward': self.total_reward,
                  'episodes': self.episode_count,
                  'episode_steps': self.episode_steps,
                  'epsilon': self.agent.epsilon
                  }

        return OrderedDict({'loss': loss, 'avg_reward': self.avg_reward,
                            'log': log, 'progress_bar': status})
