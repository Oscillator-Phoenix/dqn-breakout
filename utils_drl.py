from typing import (
    Optional,
)

import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN


class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,
            restore: Optional[str] = None,
    ) -> None:
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma

        self.__r = random.Random()
        self.__r.seed(seed)

        self.__policy = DQN(action_dim, device).to(device)
        self.__target = DQN(action_dim, device).to(device)

        if restore is None:
            self.__policy.apply(DQN.init_weights)
        else:
            self.__policy.load_state_dict(torch.load(restore))

        self.__target.load_state_dict(self.__policy.state_dict())

        self.__optimizer = optim.Adam(
            self.__policy.parameters(),
            lr=0.0001,
            eps=1.5e-4,
        )

        self.__target.eval()

    def run(self, state: TensorStack4) -> int:
        """run suggests an action for the given state."""
        with torch.no_grad():
            return self.__policy(state).max(1).indices.item()

    def run_random(self) -> int:
        '''
        return a random action
        '''
        return self.__r.randint(0, self.__action_dim - 1)

    def run_greedy(self, state: TensorStack4, epsilon: float = 1.0) -> int:
        '''
        returna a action with epsilon-greedy algorithm
        epsilon is explorate rate
        '''
        if self.__r.random() > epsilon:
            return self.run(state)
        else:
            return self.run_random()

    def run_boltzmann(self, state: TensorStack4, _lambda: float = 1.0) -> int:
        '''
        returna a action with boltzmann-exploration algorithm
        _lambda is explorate rate
        '''
        with torch.no_grad():
            probs = (
                F.softmax(self.__policy(state) * _lambda,
                          dim=-1).cpu().numpy().ravel()
            )
        return int(np.random.choice(list(range(self.__action_dim)), p=probs))

    def learn(self, memory: ReplayMemory, batch_size: int) -> float:
        """learn trains the value network via TD-learning."""

        # sampling
        state_batch, action_batch, reward_batch, next_batch, done_batch = memory.sample(
            batch_size)

        values = self.__policy(state_batch.float()).gather(1, action_batch)

        values_next = self.__target(next_batch.float()).max(1).values.detach()
        expected = (self.__gamma * values_next.unsqueeze(1)) * \
            (1. - done_batch) + reward_batch

        loss = F.smooth_l1_loss(values, expected)

        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

        return loss.item()

    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.__target.load_state_dict(self.__policy.state_dict())

    def save(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)
