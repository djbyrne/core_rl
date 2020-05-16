import argparse
from unittest import TestCase
import pytorch_lightning as pl

from algos.double_dqn.model import DoubleDQNLightning
from algos.dqn.model import DQNLightning
from algos.dueling_dqn.model import DuelingDQNLightning


class TestModels(TestCase):

    def setUp(self) -> None:
        parent_parser = argparse.ArgumentParser(add_help=False)
        parser = DQNLightning.add_model_specific_args(parent_parser)
        parser.add_argument("--algo", type=str, default="dqn", help="algorithm to use for training")
        args_list = [
            "--algo", "dqn",
            "--warm_start_steps", "500"
        ]
        self.hparams = parser.parse_args(args_list)

        self.trainer = pl.Trainer(
            gpus=0,
            max_steps=100,
            val_check_interval=1000
        )

    def test_dqn(self):
        """Smoke test that the DQN model runs"""
        model = DQNLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)

    def test_double_dqn(self):
        """Smoke test that the Double DQN model runs"""
        model = DoubleDQNLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)

    def test_dueling_dqn(self):
        """Smoke test that the Dueling DQN model runs"""
        model = DuelingDQNLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)
