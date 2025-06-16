import numpy as np
import torch

import numpy as np
import torch


class RolloutBuffer:
    """
    用于存储和管理训练数据的 Rollout Buffer 类。

    Args:
        buffer_size (int): 缓冲区的大小，即可以存储多少个时间步的数据。
        board_size (int): 棋盘的长度。
        action_dim (int): 动作空间的维度。
        gae_lambda (float): GAE (Generalized Advantage Estimation) 的 lambda 参数。
        gamma (float): 折扣因子。
        device (torch.device): 用于存储数据的设备 (CPU 或 GPU)。

    Attributes:
        buffer_size (int): 缓冲区的大小。
        board_shape (tuple): 棋盘的形状。
        action_dim (int): 动作空间的维度。
        num_channels (int): 状态表示的通道数。
        gae_lambda (float): GAE 的 lambda 参数。
        gamma (float): 折扣因子。
        device (torch.device): 用于存储数据的设备。
        states (np.ndarray): 存储状态的 NumPy 数组。
        actions (np.ndarray): 存储动作的 NumPy 数组。
        log_probs (np.ndarray): 存储动作的对数概率的 NumPy 数组。
        rewards (np.ndarray): 存储奖励的 NumPy 数组。
        values (np.ndarray): 存储状态值的 NumPy 数组。
        dones (np.ndarray): 存储 episode 是否结束的 NumPy 数组。
        advantages (np.ndarray): 存储优势估计值的 NumPy 数组。
        returns (np.ndarray): 存储回报的 NumPy 数组。
        ptr (int): 指示缓冲区中下一个可用位置的指针。
    """
    def __init__(self, buffer_size, board_size, gae_lambda, gamma, device):
        self.buffer_size = buffer_size
        self.board_shape = (board_size, board_size)
        self.action_dim = board_size * board_size
        self.num_channels = 4
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.gamma = gamma
        self.device = device

        self.states = np.zeros((buffer_size, self.num_channels, *self.board_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.bool_)

        self.advantages = np.zeros((buffer_size,), dtype=np.float32)
        self.returns = np.zeros((buffer_size,), dtype=np.float32)

        self.ptr = 0 # 缓冲区当前存储数据的指针
        self.path_start_idx = 0 # 当前轨迹段的起始索引

    def store(self, state, action, log_prob, reward, value, done):
        """
        将一个时间步的数据存储到缓冲区中。

        Args:
            state (np.ndarray): 状态。
            action (int): 动作。
            log_prob (float): 动作的对数概率。
            reward (float): 奖励。
            value (float): 状态值。
            done (bool): episode 是否结束。
        """
        if self.ptr >= self.buffer_size:
            print("Error: Rollout buffer overflow!")
            return

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done

        self.ptr += 1

    def compute_advantages(self, last_value, last_done):
        """
        计算优势估计值和回报。

        Args:
            last_value (float): 最后一个状态的值。
            last_done (bool): 最后一个状态是否是 episode 的结束状态。
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]

        advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(dones[t + 1])
                next_value = self.values[t + 1]

            # TD Error
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            # GAE 优势
            advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * advantage
            self.advantages[t + self.path_start_idx] = advantage
            # return = advantage + value
            self.returns[t + self.path_start_idx] = advantage + values[t]

        # 对当前批次的advantage进行归一化
        current_advantages = self.advantages[path_slice]
        if len(current_advantages) > 1:  # 避免单步除以零
            norm_advantages = (current_advantages - np.mean(current_advantages)) / (np.std(current_advantages) + 1e-8)
            self.advantages[path_slice] = norm_advantages
        elif len(current_advantages) == 1:
            self.advantages[self.path_start_idx] = 0

        self.path_start_idx = self.ptr

    def get_batch(self, batch_size):
        """
        生成包含批处理数据的迭代器。

        Args:
            batch_size (int): 批处理大小。

        Yields:
            tuple: 包含动作、动作的对数概率、回报、优势估计值和状态值的元组。
        """
        total_samples = self.ptr
        if total_samples == 0:
            return

        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, total_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield (
                torch.tensor(self.states[batch_indices], dtype=torch.float32).to(self.device),
                torch.tensor(self.actions[batch_indices], dtype=torch.long).to(self.device),
                torch.tensor(self.log_probs[batch_indices], dtype=torch.float32).to(self.device),
                torch.tensor(self.returns[batch_indices], dtype=torch.float32).to(self.device),
                torch.tensor(self.advantages[batch_indices], dtype=torch.float32).to(self.device),
                torch.tensor(self.values[batch_indices], dtype=torch.float32).to(self.device)
            )

    def clear(self):
        """
        清空缓冲区。
        """
        self.ptr = 0
        self.path_start_idx = 0





