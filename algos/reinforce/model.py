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
from copy import deepcopy
from itertools import chain
from typing import Tuple, List
import argparse
from collections import OrderedDict
import torch
from torch import Tensor
import torch.optim as optim
from torch.nn.functional import log_softmax
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from algos.common import wrappers
from algos.common.agents import PolicyAgent
from algos.common.memory import Experience
from algos.common.networks import MLP
from algos.dqn.core import RLDataset


class ReinforceLightning(pl.LightningModule):
    """ Basic DQN Model """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.hparams = hparams

        self.env = wrappers.make_env(self.hparams.env)
        self.env.seed(123)

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.net = None
        self.build_networks()

        self.agent = PolicyAgent(self.net)

        self.total_reward = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0

    def build_networks(self) -> None:
        """Initializes the DQN train and target networks"""
        self.net = MLP(self.obs_shape, self.n_actions)

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

    def calc_qvals(self, rewards: List[List]) -> List[List]:
        """
        Takes in the rewards for each batched episode and returns list of qvals for each batched episode

        Args:
            rewards: list of rewards for each episodes in the batch

        Returns:
            List of qvals for each episodes
        """
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r *= self.hparams.gamma
            sum_r += r
            res.append(deepcopy(sum_r))
        return list(reversed(res))

    def process_batch(self, batch: List[List[Experience]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Takes in a batch of episodes and retrieves the q vals, the states and the actions for the batch

        Args:
            batch: list of episodes, each containing a list of Experiences

        Returns:
            q_vals, states and actions used for calculating the loss
        """
        # get outputs for each episode
        batch_rewards, batch_states, batch_actions = [], [], []
        for episode in batch:
            ep_rewards, ep_states, ep_actions = [], [], []

            for step in episode:
                ep_rewards.append(step[2])
                ep_states.append(step[0])
                ep_actions.append(step[1])

            batch_rewards.append(ep_rewards)
            batch_states.append(ep_states)
            batch_actions.append(ep_actions)

        # get qvals
        batch_qvals = []
        for reward in batch_rewards:
            batch_qvals.append(self.calc_qvals(reward))

        batch_qvals = list(chain.from_iterable(batch_qvals))
        batch_states = list(chain.from_iterable(batch_states))
        batch_actions = list(chain.from_iterable(batch_actions))

        return batch_qvals, batch_states, batch_actions

    def loss(self, batch_qvals: List[Tensor], batch_states: List[Tensor], batch_actions: List[Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        # TODO: Fix shape of qvals, states, actions
        state_stack = torch.stack(batch_states)
        action_stack = torch.stack(batch_actions)

        logits = self.net(state_stack)
        log_prob = log_softmax(logits, dim=1)
        log_prob_actions = batch_qvals * log_prob[range(len(state_stack)), action_stack]
        loss = -log_prob_actions.mean()
        return loss

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
        device = self.get_device(batch)

        self.episode_reward += batch.reward
        self.episode_steps += 1

        # calculates training loss
        loss = self.loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if batch.done:
            self.total_reward = self.episode_reward
            self.episode_count += 1
            self.episode_reward = 0
            self.total_episode_steps = self.episode_steps
            self.episode_steps = 0

        log = {'total_reward': torch.tensor(self.total_reward).to(device),
               'train_loss': loss,
               'episode_steps': torch.tensor(self.total_episode_steps)
               }
        status = {'steps': torch.tensor(self.global_step).to(device),
                  'total_reward': torch.tensor(self.total_reward).to(device),
                  'episodes': self.episode_count,
                  'episode_steps': self.episode_steps
                  }

        return OrderedDict({'loss': loss, 'log': log, 'progress_bar': status})

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'

    @staticmethod
    def add_model_specific_args(parent) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model

        Note: these params are fine tuned for Pong env

        Args:
            parent
        """
        arg_parser = argparse.ArgumentParser(parents=[parent])

        arg_parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
        arg_parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        arg_parser.add_argument("--env", type=str, default="PongNoFrameskip-v4", help="gym environment tag")
        arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        arg_parser.add_argument("--sync_rate", type=int, default=1000,
                                help="how many frames do we update the target network")
        arg_parser.add_argument("--replay_size", type=int, default=100000,
                                help="capacity of the replay buffer")
        arg_parser.add_argument("--warm_start_size", type=int, default=10000,
                                help="how many samples do we use to fill our buffer at the start of training")
        arg_parser.add_argument("--eps_last_frame", type=int, default=150000,
                                help="what frame should epsilon stop decaying")
        arg_parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
        arg_parser.add_argument("--eps_end", type=float, default=0.02, help="final value of epsilon")
        arg_parser.add_argument("--episode_length", type=int, default=500, help="max length of an episode")
        arg_parser.add_argument("--max_episode_reward", type=int, default=18,
                                help="max episode reward in the environment")
        arg_parser.add_argument("--warm_start_steps", type=int, default=10000,
                                help="max episode reward in the environment")
        arg_parser.add_argument("--max_steps", type=int, default=500000,
                                help="max steps to train the agent")
        arg_parser.add_argument("--n_steps", type=int, default=4,
                                help="how many steps to unroll for each update")
        arg_parser.add_argument("--gpus", type=int, default=1,
                                help="number of gpus to use for training")
        arg_parser.add_argument("--seed", type=int, default=123,
                                help="seed for training run")
        arg_parser.add_argument("--backend", type=str, default="dp",
                                help="distributed backend to be used by lightning")
        return arg_parser
