from collections import deque
import numpy as np
import random, pdb

class ReplayBuffer():

    def __init__(self, max_size=None, min_samples_required=0):
        self.buffer = deque(maxlen=max_size)
        self.min_samples_required = min_samples_required

    def append(self, item):
        self.buffer.append(item)

    def __len__(self):
        return len(self.buffer)

    def has_enough_samples(self):
        return len(self.buffer) >= self.min_samples_required

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
