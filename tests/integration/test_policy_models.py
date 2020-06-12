import argparse
from unittest import TestCase
import pytorch_lightning as pl

from algos.double_dqn.model import DoubleDQNLightning
from algos.dqn.model import DQNLightning
from algos.dueling_dqn.model import DuelingDQNLightning
from algos.n_step_dqn.model import NStepDQNLightning
from algos.noisy_dqn.model import NoisyDQNLightning
from algos.per_dqn.model import PERDQNLightning
from algos.reinforce.model import ReinforceLightning
from algos.vanilla_policy_gradient.model import VPGLightning


class TestPolicyModels(TestCase):

    def setUp(self) -> None:
        parent_parser = argparse.ArgumentParser(add_help=False)
        parser = VPGLightning.add_model_specific_args(parent_parser)
        parser.add_argument("--algo", type=str, default="dqn", help="algorithm to use for training")
        args_list = [
            "--algo", "vpg",
            "--warm_start_steps", "100",
            "--episode_length", "100",
            "--env", "CartPole-v0"
        ]
        self.hparams = parser.parse_args(args_list)

        self.trainer = pl.Trainer(
            gpus=0,
            max_steps=100,
            max_epochs=100,  # Set this as the same as max steps to ensure that it doesn't stop early
            val_check_interval=1000  # This just needs 'some' value, does not effect training right now
        )

    def test_reinforce(self):
        """Smoke test that the DQN model runs"""
        model = ReinforceLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)

    def test_vpg(self):
        """Smoke test that the Double DQN model runs"""
        model = VPGLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)
