import argparse

import torch

from algos.dqn.model import DQNLightning
from algos.double_dqn.model import DoubleDQNLightning
import numpy as np
import pytorch_lightning as pl


def main(hparams) -> None:

    if hparams.algo is 'dqn':
        model = DQNLightning(hparams)
    elif hparams.algo is 'double_dqn':
        model = DoubleDQNLightning(hparams)

    trainer = pl.Trainer(
        gpus=1,
        distributed_backend='dp',
        early_stop_callback=False,
        val_check_interval=100
    )

    trainer.fit(model)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default='dqn', help="rl algorithm to use")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="gym environment tag")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=10, help="how many frames do we update the target network")
    parser.add_argument("--replay_size", type=int, default=1000, help="capacity of the replay buffer")
    parser.add_argument("--warm_start_size", type=int, default=1000,
                        help="how many samples do we use to fill our buffer at the start of training")
    parser.add_argument("--eps_last_frame", type=int, default=1000, help="what frame should epsilon stop decaying")
    parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
    parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
    parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
    parser.add_argument("--max_episode_reward", type=int, default=200, help="max episode reward in the environment")
    parser.add_argument("--warm_start_steps", type=int, default=1000, help="max episode reward in the environment")

    args = parser.parse_args()

    main(args)