from collections import deque
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import gym
import torch
from torch.utils.data import DataLoader

from algos.common.agents import Agent, ValueAgent
from algos.common.experience import EpisodicExperienceStream, RLDataset, ExperienceSource, NStepExperienceSource
from algos.common.memory import Experience
from algos.common.networks import MLP
from algos.common.wrappers import ToTensor


class DummyAgent(Agent):
    def __init__(self, num_envs, net):
        super().__init__(net)
        self.num_envs = num_envs

    def __call__(self, states, agent_states):
        return [0] * self.num_envs


class TestEpisodicExperience(TestCase):
    """Test the standard experience stream"""

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))
        self.net = Mock()
        self.agent = Agent(self.net)
        self.xp_stream = EpisodicExperienceStream(self.env, self.agent, device=torch.device('cpu'), episodes=4)
        self.rl_dataloader = DataLoader(self.xp_stream)

    def test_experience_stream_SINGLE_EPISODE(self):
        """Check that the experience stream gives 1 full episode per batch"""
        self.xp_stream.episodes = 1

        for i_batch, batch in enumerate(self.rl_dataloader):
            self.assertEqual(len(batch), 1)
            self.assertIsInstance(batch[0][0], Experience)
            self.assertEqual(batch[0][-1].done, True)

    def test_experience_stream_MULTI_EPISODE(self):
        """Check that the experience stream gives 4 full episodes per batch"""
        self.xp_stream.episodes = 4

        for i_batch, batch in enumerate(self.rl_dataloader):
            self.assertEqual(len(batch), 4)
            self.assertIsInstance(batch[0][0], Experience)
            self.assertEqual(batch[0][-1].done, True)


class TestExperienceSource(TestCase):

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.net = MLP(self.obs_shape, self.n_actions - 1)
        self.agent = DummyAgent(1, net=self.net)
        self.env = gym.make("CartPole-v0")
        self.source = ExperienceSource(self.env, self.agent, torch.device('cpu'))

    def test_step(self):
        """Test that outputs from source.step() provide a list of experiecnes, reward and done for each environment"""
        exp, reward, done = self.source.step()
        self.assertIsInstance(exp, list)
        self.assertIsInstance(reward, list)
        self.assertIsInstance(done, list)
        self.assertEqual(len(exp[0]), 5)
        self.assertEqual(len(reward), 1)
        self.assertEqual(len(done), 1)
        self.assertIsInstance(exp[0], Experience)
        self.assertIsInstance(reward[0], float)
        self.assertIsInstance(done[0], bool)

    def test_episode(self):
        """Test that run episode outputs the total reward for each episode"""
        total_reward = self.source.run_episode()
        self.assertIsInstance(total_reward[0], float)

    def test_multi_env(self):
        """tests that the experience source is running multiple environments"""
        self.assertIsInstance(self.source.env_pool, list)
        envs = [self.env, self.env, self.env]

        self.source = ExperienceSource(envs, self.agent, torch.device('cpu'))
        self.assertIsInstance(self.source.env_pool, list)
        self.assertEqual(len(self.source.env_pool), len(envs))
        self.assertEqual(len(self.source.states), len(envs))

    def test_multi_env_step(self):
        """tests that the experience source is running multiple environments"""
        envs = [self.env, self.env, self.env]
        agent = ValueAgent(self.net, self.n_actions)
        self.source = ExperienceSource(envs, agent, torch.device('cpu'))
        self.assertEqual(self.source.action_list, [])
        self.source.agent.epsilon = 0.0
        exps, rewards, dones = self.source.step()
        self.assertIsInstance(exps, list)
        self.assertIsInstance(rewards, list)
        self.assertIsInstance(dones, list)


class TestNStepExperienceSource(TestCase):

    def setUp(self) -> None:
        self.net = Mock()
        self.agent = DummyAgent(2, net=self.net)
        self.env = gym.make("CartPole-v0")
        self.env_pool = [gym.make("CartPole-v0"), gym.make("CartPole-v0")]
        self.n_step = 2
        self.device = torch.device('cpu')
        self.source = NStepExperienceSource(self.env_pool, self.agent, self.device, n_steps=self.n_step)

        self.state = np.zeros([32, 32])
        self.state_02 = np.ones([32, 32])
        self.next_state = np.zeros([32, 32])
        self.next_state_02 = np.ones([32, 32])
        self.action = np.zeros([1])
        self.action_02 = np.ones([1])
        self.reward = np.zeros([1])
        self.reward_02 = np.ones([1])
        self.done = np.zeros([1])
        self.done_02 = np.zeros([1])

        self.experience01 = Experience(self.state, self.action, self.reward, self.done, self.next_state)
        self.experience02 = Experience(self.state_02, self.action_02, self.reward_02, self.done_02, self.next_state_02)
        self.experience03 = Experience(self.state_02, self.action_02, self.reward_02, self.done_02, self.next_state_02)

    def test_n_step_buffer(self):
        """tests that the n_step_buffer was initialized correctly"""
        self.assertEqual(len(self.source.n_step_buffer), len(self.source.env_pool))
        self.assertIsInstance(self.source.n_step_buffer[0], deque)

    def test_step(self):
        self.assertEqual(len(self.source.n_step_buffer), len(self.env_pool))
        self.assertEqual(len(self.source.n_step_buffer[0]), 0)
        exp, reward, done = self.source.step()
        self.assertEqual(len(exp), len(self.env_pool))
        self.assertEqual(len(exp[0]), 5)
        self.assertEqual(len(self.source.n_step_buffer[0]), self.n_step)

    def test_multi_step(self):
        self.source.env_pool[0].step = Mock(return_value=(self.next_state_02, self.reward_02, self.done_02, Mock()))
        self.source.n_step_buffer.append(self.experience01)
        self.source.n_step_buffer.append(self.experience01)

        exps, rewards, dones = self.source.step()

        next_state = exps[0][4]
        self.assertEqual(next_state.all(), self.next_state_02.all())

    def test_discounted_transition(self):
        self.source = NStepExperienceSource(self.env_pool, self.agent, self.device, n_steps=3)

        for i in range(len(self.env_pool)):
            self.source.n_step_buffer[i].append(self.experience01)
            self.source.n_step_buffer[i].append(self.experience02)
            self.source.n_step_buffer[i].append(self.experience03)

            reward, next_state, done = self.source.get_transition_info(self.source.n_step_buffer[i])

            reward_01 = self.experience02.reward + 0.9 * self.experience03.reward * (1 - done)
            reward_gt = self.experience01.reward + 0.9 * reward_01 * (1 - done)

            self.assertEqual(reward, reward_gt)
            self.assertEqual(next_state.all(), self.next_state_02.all())
            self.assertEqual(self.experience03.done, done)

    def test_multi_step_discount(self):
        self.source.env_pool[0].reset = Mock(return_value=self.next_state)
        self.source.env_pool[1].reset = Mock(return_value=self.next_state)
        self.source.env_pool[0].step = Mock(return_value=(self.next_state_02, self.reward_02, self.done_02, Mock()))
        self.source.env_pool[1].step = Mock(return_value=(self.next_state_02, self.reward_02, self.done_02, Mock()))
        self.source = NStepExperienceSource(self.env_pool, self.agent, self.device, n_steps=3)

        for buffer in self.source.n_step_buffer:
            buffer.append(self.experience01)
            buffer.append(self.experience02)

        reward_gt = 1.71

        exps, rewards, dones = self.source.step()

        exp_sample = exps[0]
        self.assertEqual(exp_sample[0].all(), self.experience01.state.all())
        self.assertEqual(exp_sample[1], self.experience01.action)
        self.assertEqual(exp_sample[2], reward_gt)
        self.assertEqual(exp_sample[3], self.experience02.done)
        self.assertEqual(exp_sample[4].all(), self.experience02.new_state.all())


class TestRLDataset(TestCase):

    def setUp(self) -> None:
        mock_states = np.random.rand(32, 4, 84, 84)
        mock_action = np.random.rand(32)
        mock_rewards = np.random.rand(32)
        mock_dones = np.random.rand(32)
        mock_next_states = np.random.rand(32, 4, 84, 84)
        self.sample = mock_states, mock_action, mock_rewards, mock_dones, mock_next_states

        self.buffer = Mock()
        self.buffer.sample = Mock(return_value=self.sample)
        self.dataset = RLDataset(buffer=self.buffer, sample_size=32)
        self.dl = DataLoader(self.dataset, batch_size=32)

    def test_rl_dataset_batch(self):
        """test that the dataset gives the correct batch"""

        for i_batch, sample_batched in enumerate(self.dl):
            self.assertIsInstance(sample_batched, list)
            self.assertEqual(sample_batched[0].shape, torch.Size([32, 4, 84, 84]))
            self.assertEqual(sample_batched[1].shape, torch.Size([32]))
            self.assertEqual(sample_batched[2].shape, torch.Size([32]))
            self.assertEqual(sample_batched[3].shape, torch.Size([32]))
            self.assertEqual(sample_batched[4].shape, torch.Size([32, 4, 84, 84]))



