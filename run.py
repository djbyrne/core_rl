"""Run script for training agents"""

import argparse
import random

import torch

from algos.dqn.model import DQNLightning
from algos.double_dqn.model import DoubleDQNLightning
import numpy as np
import pytorch_lightning as pl

from algos.dueling_dqn.model import DuelingDQNLightning
from algos.n_step_dqn.model import NStepDQNLightning
from algos.noisy_dqn.model import NoisyDQNLightning
from algos.per_dqn.model import PERDQNLightning
from algos.reinforce.model import ReinforceLightning
from algos.vanilla_policy_gradient.model import VPGLightning


def main(hparams) -> None:
    """Main Training Method"""

    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    # TODO: this is shit, fix soon
    if hparams.algo == 'double_dqn':
        model = DoubleDQNLightning(hparams)
    elif hparams.algo == 'dueling_dqn':
        model = DuelingDQNLightning(hparams)
    elif hparams.algo == 'noisy_dqn':
        model = NoisyDQNLightning(hparams)
    elif hparams.algo == 'per_dqn':
        model = PERDQNLightning(hparams)
    elif hparams.algo == 'n_step_dqn':
        model = NStepDQNLightning(hparams)
    elif hparams.algo == 'reinforce':
        model = ReinforceLightning(hparams)
    elif hparams.algo == 'vpg':
        model = VPGLightning(hparams)
    else:
        model = DQNLightning(hparams)

    trainer = pl.Trainer(
        gpus=hparams.gpus,
        distributed_backend=hparams.backend,
        max_steps=10000,
        max_epochs=hparams.max_steps,       # Set this as the same as max steps to ensure that it doesn't stop early
        val_check_interval=1000,             # This just needs 'some' value, does not effect training right now
        profiler=True
    )

    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    parser = DQNLightning.add_model_specific_args(parent_parser)
    parser.add_argument("--algo", type=str, default="dqn", help="algorithm to use for training")
    args = parser.parse_args()

    main(args)
