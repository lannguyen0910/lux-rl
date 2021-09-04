from collections import namedtuple, deque
import random
import torch
import numpy as np


class ReplayBuffer:
    """
    A cyclic buffer of bounded size that holds the transitions observed recently
    """

    def __init__(self, cfg):
        self.memory = deque(maxlen=cfg.buffer_size)
        self.batch_size = cfg.batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "new_state", "reward", "done"])
        self.seed = random.seed(cfg.seed)
        self.device = cfg.device

    def add(self, state, action, new_state, reward, done):
        e = self.experience(state=state, action=action,
                            new_state=new_state, reward=reward, done=done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states1 = torch.from_numpy(np.stack(
            [e.state for e in experiences if e is not None], axis=0)).float().to(self.device)
        actions = torch.from_numpy(np.stack(
            [e.action for e in experiences if e is not None], axis=0)).float().to(self.device)
        states2 = torch.from_numpy(np.stack(
            [e.new_state for e in experiences if e is not None], axis=0)).float().to(self.device)
        rewards = torch.from_numpy(np.stack(
            [e.reward for e in experiences if e is not None], axis=0)).float().to(self.device)
        dones = torch.from_numpy(np.stack(
            [e.done for e in experiences if e is not None], axis=0)).float().to(self.device)

        return (states1, actions, states2, rewards, dones)

    def __len__(self):
        return len(self.memory)
