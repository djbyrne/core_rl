"""Series of memory buffers sued"""

# Named tuple for storing experience steps gathered in training
from typing import Tuple, List
from collections import deque, namedtuple
import numpy as np

Experience = namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        """
        Takes a sample of the buffer
        Args:
            batch_size: current batch_size

        Returns:
            a batch of tuple np arrays of Experiences
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))


class PERBuffer:
    """simple list based Prioritized Experience Replay Buffer"""

    def __init__(self, buffer_size, prob_alpha=0.6, beta_start=0.4, beta_frames=100000):
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

    def __len__(self):
        return len(self.buffer)

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
