from unittest import TestCase

import numpy
import torch

from algos.common.memory import ReplayBuffer, Experience, PERBuffer, MultiStepBuffer


class TestReplayBuffer(TestCase):

    def setUp(self) -> None:
        self.buffer = ReplayBuffer(10)

        self.state = numpy.random.rand(32, 32)
        self.next_state = numpy.random.rand(32, 32)
        self.action = numpy.ones([1])
        self.reward = numpy.ones([1])
        self.done = numpy.zeros([1])
        self.experience = Experience(self.state, self.action, self.reward, self.done, self.next_state)

    def test_replay_buffer_APPEND(self):
        """Test that you can append to the replay buffer"""

        self.assertEqual(len(self.buffer), 0)

        self.buffer.append(self.experience)

        self.assertEqual(len(self.buffer), 1)

    def test_replay_buffer_SAMPLE(self):
        """Test that you can sample from the buffer and the outputs are the correct shape"""
        batch_size = 3

        for i in range(10):
            self.buffer.append(self.experience)

        batch = self.buffer.sample(batch_size)

        self.assertEqual(len(batch), 5)

        # states
        states = batch[0]
        self.assertEqual(states.shape, (batch_size, 32, 32))
        # action
        actions = batch[1]
        self.assertEqual(actions.shape, (batch_size, 1))
        # reward
        rewards = batch[2]
        self.assertEqual(rewards.shape, (batch_size, 1))
        # dones
        dones = batch[3]
        self.assertEqual(dones.shape, (batch_size, 1))
        # next states
        next_states = batch[4]
        self.assertEqual(next_states.shape, (batch_size, 32, 32))


class TestPrioReplayBuffer(TestCase):

    def setUp(self) -> None:
        self.buffer = PERBuffer(10)

        self.state = numpy.random.rand(32, 32)
        self.next_state = numpy.random.rand(32, 32)
        self.action = numpy.ones([1])
        self.reward = numpy.ones([1])
        self.done = numpy.zeros([1])
        self.experience = Experience(self.state, self.action, self.reward, self.done, self.next_state)

    def test_replay_buffer_APPEND(self):
        """Test that you can append to the replay buffer and the latest experience has max priority"""

        self.assertEqual(len(self.buffer), 0)

        self.buffer.append(self.experience)

        self.assertEqual(len(self.buffer), 1)
        self.assertEqual(self.buffer.priorities[0], 1.0)

    def test_replay_buffer_SAMPLE(self):
        """Test that you can sample from the buffer and the outputs are the correct shape"""
        batch_size = 3

        for i in range(10):
            self.buffer.append(self.experience)

        batch, indices, weights = self.buffer.sample(batch_size)

        self.assertEqual(len(batch), 5)
        self.assertEqual(len(indices), batch_size)
        self.assertEqual(len(weights), batch_size)


        # states
        states = batch[0]
        self.assertEqual(states.shape, (batch_size, 32, 32))
        # action
        actions = batch[1]
        self.assertEqual(actions.shape, (batch_size, 1))
        # reward
        rewards = batch[2]
        self.assertEqual(rewards.shape, (batch_size, 1))
        # dones
        dones = batch[3]
        self.assertEqual(dones.shape, (batch_size, 1))
        # next states
        next_states = batch[4]
        self.assertEqual(next_states.shape, (batch_size, 32, 32))


class TestMultiStepReplayBuffer(TestCase):

    def setUp(self) -> None:
        self.buffer = MultiStepBuffer(buffer_size=10, n_step=2)

        self.state = numpy.zeros([32, 32])
        self.state_02 = numpy.ones([32, 32])
        self.next_state = numpy.zeros([32, 32])
        self.next_state_02 = numpy.ones([32, 32])
        self.action = numpy.zeros([1])
        self.action_02 = numpy.ones([1])
        self.reward = numpy.zeros([1])
        self.reward_02 = numpy.ones([1])
        self.done = numpy.zeros([1])
        self.done_02 = numpy.zeros([1])

        self.experience01 = Experience(self.state, self.action, self.reward, self.done, self.next_state)
        self.experience02 = Experience(self.state_02, self.action_02, self.reward_02, self.done_02, self.next_state_02)
        self.experience03 = Experience(self.state_02, self.action_02, self.reward_02, self.done_02, self.next_state_02)

    def test_append_single_experience_LESS_THAN_N(self):
        """
        If a single experience is added and n > 1 nothing should be added to the buffer as it is waiting experiences
        to equal n
        """
        self.assertEqual(len(self.buffer), 0)

        self.buffer.append(self.experience01)

        self.assertEqual(len(self.buffer), 0)

    def test_append_single_experience(self):
        """
        If a single experience is added and n > 1 nothing should be added to the buffer as it is waiting experiences
        to equal n
        """
        self.assertEqual(len(self.buffer), 0)

        self.buffer.append(self.experience01)

        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(len(self.buffer.n_step_buffer), 1)

    def test_append_single_experience(self):
        """
        If a single experience is added and the number of experiences collected >= n, the multi step experience should
        be added to the full buffer.
        """
        self.assertEqual(len(self.buffer), 0)

        self.buffer.append(self.experience01)
        self.buffer.append(self.experience02)

        self.assertEqual(len(self.buffer), 1)
        self.assertEqual(len(self.buffer.n_step_buffer), 2)

    def test_sample_single_experience(self):
        """if there is only a single experience added, sample should return nothing"""
        self.buffer.append(self.experience01)

        with self.assertRaises(Exception) as context:
            _ = self.buffer.sample(batch_size=1)

        self.assertIsInstance(context.exception, Exception)

    def test_sample_multi_experience(self):
        """if there is only a single experience added, sample should return nothing"""
        self.buffer.append(self.experience01)
        self.buffer.append(self.experience02)

        batch = self.buffer.sample(batch_size=1)

        next_state = batch[4]
        self.assertEqual(next_state.all(), self.next_state_02.all())

    def test_get_transition_info_2_STEP(self):
        """Test that the accumulated experience is correct and"""
        self.buffer.append(self.experience01)
        self.buffer.append(self.experience02)

        reward, next_state, done = self.buffer.get_transition_info()

        reward_gt = self.experience01.reward + (0.9 * self.experience02.reward) * (1 - done)

        self.assertEqual(reward, reward_gt)
        self.assertEqual(next_state.all(), self.next_state_02.all())
        self.assertEqual(self.experience02.done, done)

    def test_get_transition_info_3_STEP(self):
        """Test that the accumulated experience is correct with multi step"""
        self.buffer = MultiStepBuffer(buffer_size=10, n_step=3)

        self.buffer.append(self.experience01)
        self.buffer.append(self.experience02)
        self.buffer.append(self.experience02)

        reward, next_state, done = self.buffer.get_transition_info()

        reward_01 = self.experience02.reward + 0.9 * self.experience03.reward * (1 - done)
        reward_gt = self.experience01.reward + 0.9 * reward_01 * (1 - done)

        self.assertEqual(reward, reward_gt)
        self.assertEqual(next_state.all(), self.next_state_02.all())
        self.assertEqual(self.experience03.done, done)

    def test_sampele_3_STEP(self):
        self.buffer = MultiStepBuffer(buffer_size=10, n_step=3)

        self.buffer.append(self.experience01)
        self.buffer.append(self.experience02)
        self.buffer.append(self.experience02)

        reward_gt = 1.71

        batch = self.buffer.sample(1)

        self.assertEqual(batch[0].all(), self.experience01.state.all())
        self.assertEqual(batch[1], self.experience01.action)
        self.assertEqual(batch[2], reward_gt)
        self.assertEqual(batch[3], self.experience02.done)
        self.assertEqual(batch[4].all(), self.experience02.new_state.all())
