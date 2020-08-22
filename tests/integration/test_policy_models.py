import argparse
from unittest import TestCase

import pytorch_lightning as pl

from algos.common import cli
from algos.reinforce.model import Reinforce
from algos.vanilla_policy_gradient.model import PolicyGradient


class TestPolicyModels(TestCase):

    def setUp(self) -> None:
        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser = cli.add_base_args(parent=parent_parser)
        parent_parser = PolicyGradient.add_model_specific_args(parent_parser)
        args_list = [
            "--algo", "PolicyGradient",
            "--episode_length", "100",
            "--env", "CartPole-v0"
        ]
        self.hparams = parent_parser.parse_args(args_list)

        self.trainer = pl.Trainer(
            gpus=0,
            max_steps=100,
            max_epochs=100,  # Set this as the same as max steps to ensure that it doesn't stop early
            fast_dev_run=True
        )

    def test_reinforce(self):
        """Smoke test that the reinforce model runs"""

        model = Reinforce(self.hparams.env)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)

    def test_PolicyGradient(self):
        """Smoke test that the policy gradient model runs"""
        model = PolicyGradient(self.hparams.env)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)
