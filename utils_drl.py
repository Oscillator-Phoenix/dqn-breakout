from typing import (
    Optional,
)


import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from pfrl import replay_buffers

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN, DuelingDQN


class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,
            restore: Optional[str] = None,
            q_func: Optional[str] = None
    ) -> None:
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma

        self.__r = random.Random()
        self.__r.seed(seed)

        if q_func is None:
            self.__policy = DQN(action_dim, device).to(device)
            self.__target = DQN(action_dim, device).to(device)
        elif q_func == "DuelingDQN":
            self.__policy = DuelingDQN(action_dim, device).to(device)
            self.__target = DuelingDQN(action_dim, device).to(device)
        else:
            raise NotImplementedError

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

    # ---------------------------------------

    def learn_prioritized(
            self, memory: replay_buffers.PrioritizedReplayBuffer, batch_size: int) -> float:
        """
        learn trains the value network via TD-learning with Prioritized Experience Replay
        """

        # 从 Prioritized Experience Replay 抽取带 weight 的 experiences
        experiences = memory.sample(batch_size)
        state_batch = torch.cat(
            [elem[0]["state"] for elem in experiences]).to(self.__device).float()
        action_batch = torch.tensor(
            [elem[0]["action"] for elem in experiences]).reshape(batch_size, 1).to(self.__device)
        reward_batch = torch.tensor(
            [elem[0]["reward"] for elem in experiences]).reshape(batch_size, 1).to(self.__device).float()
        done_batch = torch.tensor(
            [elem[0]["is_state_terminal"] for elem in experiences]).reshape(batch_size, 1).to(self.__device).float()
        next_batch = torch.cat(
            [elem[0]["next_state"] for elem in experiences]).to(self.__device).float()
        weight_batch = torch.tensor(
            [elem[0]["weight"] for elem in experiences]).reshape(batch_size, 1).to(self.__device).float()

        # 计算 y 和 target
        # y, t = self._compute_y_and_t(exp_batch)
        values = self.__policy(state_batch.float()).gather(1, action_batch)

        with torch.no_grad():
            values_next = self.__target(next_batch.float()).max(1).values
            expected = reward_batch + \
                self.__gamma * (1. - done_batch) * values_next.unsqueeze(1)

        # print("\n state_batch shape", state_batch.size())
        # print("\n action_batch shape", action_batch.size())
        # print("\n reward_batch shape", reward_batch.size())
        # print("\n done_batch shape", done_batch.size())
        # print("\n next_batch shape", next_batch.size())
        # print("\n weight_batch shape", weight_batch.size())
        # print("\n values shape", values.size())
        # print("\n expected shape", expected.size())
        # print("value", values)
        # print("expected", expected)
        # print("\n q(s, _ ) shape", self.__policy(state_batch.float()).size())

        # y, t
        y = values.reshape(-1, 1)   # y
        t = expected.reshape(-1, 1)  # target

        # 通过 y 和 t 计算 delta，用于更新 Experience 的 priority
        errors_out = []
        delta = torch.abs(y - t)
        if delta.ndim == 2:
            delta = torch.sum(delta, dim=1)
        delta = delta.detach().cpu().numpy()
        for e in delta:
            errors_out.append(e)

        # 使用 errors_out 更新 Experience 的 priority
        memory.update_errors(errors_out)

        # 通过 y t weight 计算 loss
        loss_batch = F.smooth_l1_loss(y, t, reduction="none")
        loss_batch.reshape(-1,)
        weight_batch.reshape(-1,)
        loss_sum = torch.sum(loss_batch * weight_batch)
        loss = loss_sum / y.shape[0]
        # HERE: You can record loss

        # print("\n y shape", y.size())
        # print("\n t shape", t.size())
        # print("\n loss_batch shape", loss_batch.size())
        # print("\n weight_batch shape", weight_batch.size())
        # print("\n loss", loss.size(), loss)

        # loss 反向传播， 梯度更新网络
        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

# ---------------------------------------------------------------------------
