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

import pytorch_lightning as pl

from typing import Tuple, List

import argparse
from collections import OrderedDict, deque, namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from algos.common import wrappers
from algos.common.memory import ReplayBuffer
from algos.common.networks import MLP, CNN
from algos.dqn.core import RLDataset, Agent


class DoubleDQNLightning(pl.LightningModule):
    """ Double DQN Model """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.hparams = hparams

        self.env = wrappers.make_env(self.hparams.env)
        self.env.seed(123)

        obs_shape = self.env.observation_space.shape
        n_actions = self.env.action_space.n

        self.net = CNN(obs_shape, n_actions)
        self.target_net = CNN(obs_shape, n_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer. This uses an improvement to the original
        DQN loss by using the double dqn. This is shown by using the actions of the train network to pick the
        value from the target network. This code is heavily commented in order to explain the process clearly

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch  # batch of experiences, batch_size = 16

        actions_v = actions.unsqueeze(-1)  # adds a dimension, 16 -> [16, 1]
        output = self.net(states)  # shape [16, 2], [batch, action space]

        # gather the value of the outputs according to the actions index from the batch
        state_action_values = output.gather(1, actions_v).squeeze(-1)

        # dont want to mess with gradients when using the target network
        with torch.no_grad():
            next_outputs = self.net(next_states)  # [16, 2], [batch, action_space]

            next_state_acts = next_outputs.max(1)[1].unsqueeze(-1)   # take action at the index with the highest value
            next_tgt_out = self.target_net(next_states)

            # Take the value of the action chosen by the train network
            next_state_values = next_tgt_out.gather(1, next_state_acts).squeeze(-1)
            next_state_values[dones] = 0.0  # any steps flagged as done get a 0 value
            next_state_values = next_state_values.detach()  # remove values from the graph, no grads needed

        # calc expected discounted return of next_state_values
        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        # Standard MSE loss between the state action values of the current state and the
        # expected state action values of the next state
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(self.hparams.eps_end, self.hparams.eps_start -
                      (self.global_step + 1) / self.hparams.eps_last_frame)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward
        self.episode_steps += 1

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_count += 1
            self.episode_reward = 0
            self.total_episode_steps = self.episode_steps
            self.episode_steps = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {'total_reward': torch.tensor(self.total_reward).to(device),
               'reward': torch.tensor(reward).to(device),
               'train_loss': loss,
               'episode_steps': torch.tensor(self.total_episode_steps)
               }
        status = {'steps': torch.tensor(self.global_step).to(device),
                  'total_reward': torch.tensor(self.total_reward).to(device),
                  'episodes': self.episode_count,
                  'episode_steps': self.episode_steps,
                  'epsilon': epsilon
                  }

        return OrderedDict({'loss': loss, 'log': log, 'progress_bar': status})

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'


def main(hparams) -> None:
    model = DoubleDQNLightning(hparams)

    trainer = pl.Trainer(
        gpus=1,
        distributed_backend='dp',
        max_epochs=10000,
        val_check_interval=100
    )

    trainer.fit(model)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="gym environment tag")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=1000,
                        help="how many frames do we update the target network")
    parser.add_argument("--replay_size", type=int, default=10000,
                        help="capacity of the replay buffer")
    parser.add_argument("--warm_start_size", type=int, default=10000,
                        help="how many samples do we use to fill our buffer at the start of training")
    parser.add_argument("--eps_last_frame", type=int, default=150000,
                        help="what frame should epsilon stop decaying")
    parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
    parser.add_argument("--eps_end", type=float, default=0.02, help="final value of epsilon")
    parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
    parser.add_argument("--max_episode_reward", type=int, default=21,
                        help="max episode reward in the environment")
    parser.add_argument("--warm_start_steps", type=int, default=10000,
                        help="max episode reward in the environment")

    args, _ = parser.parse_known_args()

    main(args)