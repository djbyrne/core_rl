"""Run script for training agents"""

import argparse
import random

import torch

from algos.dqn.model import DQNLightning
from algos.double_dqn.model import DoubleDQNLightning
import numpy as np
import pytorch_lightning as pl

from algos.dueling_dqn.model import DuelingDQNLightning


def main(hparams) -> None:
    """Main Training Method"""

    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    if hparams.algo is 'double_dqn':
        model = DoubleDQNLightning(hparams)
    elif hparams.algo is 'dueling_dqn':
        model = DuelingDQNLightning(hparams)
    else:
        model = DQNLightning(hparams)

    trainer = pl.Trainer(
        gpus=hparams.gpus,
        distributed_backend=hparams.backend,
        max_steps=hparams.max_steps,
        val_check_interval=1000
    )

    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    parser = DQNLightning.add_model_specific_args(parent_parser)
    parser.add_argument("--algo", type=str, default="dqn", help="algorithm to use for training")
    args = parser.parse_args()

    main(args)
