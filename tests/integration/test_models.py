import argparse
from unittest import TestCase
import pytorch_lightning as pl

from algos.double_dqn.model import DoubleDQNLightning
from algos.dqn.model import DQNLightning
from algos.dueling_dqn.model import DuelingDQNLightning
from algos.n_step_dqn.model import NStepDQNLightning
from algos.noisy_dqn.model import NoisyDQNLightning
from algos.per_dqn.model import PERDQNLightning



class TestModels(TestCase):

    def setUp(self) -> None:
        parent_parser = argparse.ArgumentParser(add_help=False)
        parser = DQNLightning.add_model_specific_args(parent_parser)
        parser.add_argument("--algo", type=str, default="dqn", help="algorithm to use for training")
        args_list = [
            "--algo", "dqn",
            "--warm_start_steps", "100",
            "--episode_length", "100"
        ]
        self.hparams = parser.parse_args(args_list)

        self.trainer = pl.Trainer(
            gpus=0,
            max_steps=100,
            max_epochs=100,  # Set this as the same as max steps to ensure that it doesn't stop early
            val_check_interval=1000  # This just needs 'some' value, does not effect training right now
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

    def test_noisy_dqn(self):
        """Smoke test that the Noisy DQN model runs"""
        model = NoisyDQNLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)

    def test_per_dqn(self):
        """Smoke test that the PER DQN model runs"""
        model = PERDQNLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)

    def test_n_step_dqn(self):
        """Smoke test that the N Step DQN model runs"""
        model = NStepDQNLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)
