"""
Noisy Deep Reinforcement Learning
"""

from algos.common.networks import NoisyCNN
from algos.dqn.model import DQNLightning


class NoisyDQNLightning(DQNLightning):
    """ Noisy DQN Model """

    def build_networks(self) -> None:
        """Initializes the Noisy DQN train and target networks"""
        self.net = NoisyCNN(self.obs_shape, self.n_actions)
        self.target_net = NoisyCNN(self.obs_shape, self.n_actions)

    def on_train_start(self) -> None:
        """Set the agents epsilon to 0 as the exploration comes from the network"""
        self.agent.epsilon = 0.0
