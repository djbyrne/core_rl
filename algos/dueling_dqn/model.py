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
from algos.common.networks import DuelingCNN
from algos.dqn.model import DQNLightning


class DuelingDQNLightning(DQNLightning):
    """ Dueling DQN Model """

    def build_networks(self) -> None:
        """Initializes the Dueling DQN train and target networks"""
        self.net = DuelingCNN(self.obs_shape, self.n_actions)
        self.target_net = DuelingCNN(self.obs_shape, self.n_actions)


def main(hparams) -> None:
    """Main Run Method"""
    model = DQNLightning(hparams)

    trainer = pl.Trainer(
        gpus=hparams.gpus,
        distributed_backend='dp',
        max_epochs=hparams.max_epochs,
        val_check_interval=1000
    )

    trainer.fit(model)
