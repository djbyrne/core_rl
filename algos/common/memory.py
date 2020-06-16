"""Series of memory buffers sued"""

# Named tuple for storing experience steps gathered in training
import collections
from typing import Tuple, List, Union
from collections import deque, namedtuple

import numpy as np
import torch

Experience = namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class Buffer:
    """
    Basic Buffer for storing a single experience at a time

    Args:
        capacity: size of the buffer
    """
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    @torch.no_grad
    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    # pylint: disable=unused-argument
    def sample(self, *args) -> Union[Tuple, List[Tuple]]:
        """
        returns everything in the buffer so far it is then reset

        Returns:
            a batch of tuple np arrays of state, action, reward, done, next_state
        """
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in range(self.__len__())])

        self.buffer.clear()

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))


class ReplayBuffer(Buffer):
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them

    Args:
        capacity: size of the buffer
    """

    def sample(self, batch_size: int) -> Tuple:
        """
        Takes a sample of the buffer
        Args:
            batch_size: current batch_size

        Returns:
            a batch of tuple np arrays of state, action, reward, done, next_state
        """

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))


class MultiStepBuffer:
    """
    N Step Replay Buffer

    Deprecated: use the NStepExperienceSource with the standard ReplayBuffer
    """
    def __init__(self, buffer_size, n_step=2):
        self.n_step = n_step
        self.buffer = deque(maxlen=buffer_size)
        self.n_step_buffer = deque(maxlen=n_step)

    def __len__(self):
        return len(self.buffer)

    def get_transition_info(self, gamma=0.9) -> Tuple[np.float, np.array, np.int]:
        """
        get the accumulated transition info for the n_step_buffer
        Args:
            gamma: discount factor
        Returns:
            multi step reward, final observation and done
        """
        last_experience = self.n_step_buffer[-1]
        final_state = last_experience.new_state
        done = last_experience.done
        reward = last_experience.reward

        # calculate reward
        # in reverse order, go through all the experiences up till the first experience
        for experience in reversed(list(self.n_step_buffer)[:-1]):
            reward_t = experience.reward
            new_state_t = experience.new_state
            done_t = experience.done

            reward = reward_t + gamma * reward * (1 - done_t)
            final_state, done = (new_state_t, done_t) if done_t else (final_state, done)

        return reward, final_state, done

    @torch.no_grad
    def append(self, experience) -> None:
        """
        add an experience to the buffer by collecting n steps of experiences
        Args:
            experience: tuple (state, action, reward, done, next_state)
        """
        self.n_step_buffer.append(experience)

        if len(self.n_step_buffer) >= self.n_step:
            reward, next_state, done = self.get_transition_info()
            first_experience = self.n_step_buffer[0]
            multi_step_experience = Experience(first_experience.state,
                                               first_experience.action,
                                               reward,
                                               done,
                                               next_state)

            self.buffer.append(multi_step_experience)

    def sample(self, batch_size: int) -> Tuple:
        """
        Takes a sample of the buffer
        Args:
            batch_size: current batch_size
        Returns:
            a batch of tuple np arrays of Experiences
        """
        # pylint: disable=no-else-return
        if len(self.buffer) >= batch_size:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

            return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                    np.array(dones, dtype=np.bool), np.array(next_states))
        else:
            raise Exception('Buffer length is less than the batch size')


class MeanBuffer:
    """
    Stores a deque of items and calculates the mean
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.deque = collections.deque(maxlen=capacity)
        self.sum = 0.0

    def add(self, val: float) -> None:
        """Add to the buffer"""
        if len(self.deque) == self.capacity:
            self.sum -= self.deque[0]
        self.deque.append(val)
        self.sum += val

    def mean(self) -> float:
        """Retrieve the mean"""
        if not self.deque:
            return 0.0
        return self.sum / len(self.deque)


class PERBuffer(ReplayBuffer):
    """simple list based Prioritized Experience Replay Buffer"""

    def __init__(self, buffer_size, prob_alpha=0.6, beta_start=0.4, beta_frames=100000):
        super().__init__(capacity=buffer_size)
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.prob_alpha = prob_alpha
        self.capacity = buffer_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)

    def update_beta(self, step) -> float:
        """
        Update the beta value which accounts for the bias in the PER

        Args:
            step: current global step

        Returns:
            beta value for this indexed experience
        """
        beta_val = self.beta_start + step * (1.0 - self.beta_start) / self.beta_frames
        self.beta = min(1.0, beta_val)

        return self.beta

    @torch.no_grad
    def append(self, exp) -> None:
        """
        Adds experiences from exp_source to the PER buffer

        Args:
            exp: experience tuple being added to the buffer
        """
        # what is the max priority for new sample
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp

        # the priority for the latest sample is set to max priority so it will be resampled soon
        self.priorities[self.pos] = max_prio

        # update position, loop back if it reaches the end
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=32) -> Tuple:
        """
        Takes a prioritized sample from the buffer

        Args:
            batch_size: size of sample

        Returns:
            sample of experiences chosen with ranked probability
        """
        # get list of priority rankings
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # probability to the power of alpha to weight how important that probability it, 0 = normal distirbution
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        # choise sample of indices based on the priority prob distribution
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        # samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        samples = (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                   np.array(dones, dtype=np.bool), np.array(next_states))
        total = len(self.buffer)

        # weight of each sample datum to compensate for the bias added in with prioritising samples
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # return the samples, the indices chosen and the weight of each datum in the sample
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices: List, batch_priorities: List) -> None:
        """
        Update the priorities from the last batch, this should be called after the loss for this batch has been
        calculated.

        Args:
            batch_indices: index of each datum in the batch
            batch_priorities: priority of each datum in the batch
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

#
# class SumTree:
#     write = 0
#
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.tree = np.zeros(2 * capacity - 1)
#         self.data = np.zeros(capacity, dtype=object)
#         self.n_entries = 0
#
#     # update to the root node
#     def _propagate(self, idx, change):
#         parent = (idx - 1) // 2
#
#         self.tree[parent] += int(change)
#
#         if parent != 0:
#             self._propagate(parent, change)
#
#     # find sample on leaf node
#     def _retrieve(self, idx, s):
#         left = 2 * idx + 1
#         right = left + 1
#
#         if left >= len(self.tree):
#             return idx
#
#         if s <= self.tree[left]:
#             return self._retrieve(left, s)
#         else:
#             return self._retrieve(right, s - self.tree[left])
#
#     def total(self):
#         return self.tree[0]
#
#     # store priority and sample
#     def add(self, p, data):
#         idx = self.write + self.capacity - 1
#
#         self.data[self.write] = data
#         self.update(idx, p)
#
#         self.write += 1
#         if self.write >= self.capacity:
#             self.write = 0
#
#         if self.n_entries < self.capacity:
#             self.n_entries += 1
#
#     # update priority
#     def update(self, idx, p):
#         change = p - self.tree[idx]
#
#         self.tree[idx] = p
#
#         if isinstance(idx, int):
#             self._propagate(idx, change)
#         else:
#             for index, node in enumerate(idx):
#                 self._propagate(node, change[index])
#
#     # get priority and sample
#     def get(self, s):
#         idx = self._retrieve(0, s)
#         dataIdx = idx - self.capacity + 1
#
#         return (idx, self.tree[idx], self.data[dataIdx])
#
#
# class SumTreeBuffer:  # stored as ( s, a, r, s_ ) in SumTree
#     e = 0.01
#     a = 0.6
#     beta = 0.4
#     beta_increment_per_sampling = 0.001
#
#     def __init__(self, capacity):
#         self.tree = SumTree(capacity)
#         self.capacity = capacity
#         self.step = 0
#         self.beta_start = 0.4
#         self.beta_frames = 100000
#
#     def _get_priority(self, error):
#         return (np.abs(error)) ** self.a
#
#     def append(self, sample):
#         # p = self._get_priority(error)
#         p = 1.0
#         self.tree.add(p, sample)
#
#     def sample(self, n):
#         self.step += 1
#         batch = []
#         idxs = []
#         segment = self.tree.total() / n
#         priorities = []
#
#         # self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
#         beta_val = self.beta_start + self.step * (1.0 - self.beta_start) / self.beta_frames
#         self.beta = min(1.0, beta_val)
#
#         for i in range(n):
#             a = segment * i
#             b = segment * (i + 1)
#
#             s = random.uniform(a, b)
#             (idx, p, data) = self.tree.get(s)
#             priorities.append(p)
#             sample = (np.array(data.state), np.array(data.action), np.array(data.reward, dtype=np.float32),
#                       np.array(data.done, dtype=np.bool), np.array(data.new_state))
#             batch.append(sample)
#             idxs.append(idx)
#
#         sampling_probabilities = priorities / self.tree.total()
#         is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
#         is_weight /= is_weight.max()
#
#         return batch, idxs, is_weight
#
#     def update_priorities(self, idx, error):
#         p = self._get_priority(error)
#         self.tree.update(idx, p)
#
