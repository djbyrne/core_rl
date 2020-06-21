"""Run script for training agents"""

import argparse
import random
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl

from algos.common import cli
from algos.dqn.model import DQNLightning
from algos.double_dqn.model import DoubleDQNLightning
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

    # TODO: make this better
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

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='total_reward',
        mode='max',
        prefix=''
    )

    trainer = pl.Trainer(
        gpus=hparams.gpus,
        distributed_backend=hparams.backend,
        max_steps=hparams.max_steps,
        max_epochs=hparams.max_steps,
        val_check_interval=1000,
        profiler=True,
        checkpoint_callback=checkpoint_callback,
    )

    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = cli.add_base_args(parent=parent_parser)
    parent_parser = DQNLightning.add_model_specific_args(parent_parser)
    parent_parser = VPGLightning.add_model_specific_args(parent_parser)

    args = parent_parser.parse_args()

    main(args)
