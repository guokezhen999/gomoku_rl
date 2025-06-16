import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from agents.net import PolicyValueNet
from agents.buffer import RolloutBuffer
from agents.basic import Agent

class PPOAgent(Agent):
    """
    近端策略优化（Proximal Policy Optimization，PPO）代理。

    Args:
        board_size (int): 棋盘长度。
        gamma (float): 折扣因子。
        gae_lambda (float): 广义优势估计（Generalized Advantage Estimation，GAE）的 lambda 参数。
        clip_epsilon (float): PPO 裁剪参数。
        ppo_epochs (int): PPO 更新的 epoch 数量。
        num_mini_batches (int): 用于 PPO 更新的 mini-batch 数量。
        entropy_coef (float): 熵损失系数。
        value_loss_coef (float): 价值损失系数。
        device (torch.device): 用于训练的设备（CPU 或 GPU）。
        optimizer_params(dict): 优化器需要参数。

    Attributes:
        num_channels (int): 输入状态的通道数（例如，对于井字棋，通常为 4）。
        board_shape (tuple): 棋盘的形状（长度，长度）。
        action_dim (int): 动作维度（棋盘长度 * 棋盘长度）。
        gamma (float): 折扣因子。
        gae_lambda (float): GAE 的 lambda 参数。
        clip_epsilon (float): PPO 裁剪参数。
        ppo_epochs (int): PPO 更新的 epoch 数量。
        num_mini_batches (int): mini-batch 数量。
        entropy_coef (float): 熵损失系数。
        value_loss_coef (float): 价值损失系数。
        device (torch.device): 用于训练的设备。
        net (PolicyValueNet): 策略价值网络。
        optimizer (torch.optim.Adam): Adam 优化器。
        mse_loss (torch.nn.MSELoss): 均方误差损失函数。
    """
    def __init__(self, board_size, gamma, gae_lambda, clip_epsilon, ppo_epochs, num_mini_batches,
                 entropy_coef, value_loss_coef, action_device, update_device, optimizer_params):
        super().__init__(board_size)
        self.num_channels = 4
        self.board_shape = (board_size, board_size)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.num_mini_batches = num_mini_batches
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.action_device = action_device
        self.update_device = update_device

        self.net = PolicyValueNet(board_size).to(self.action_device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=optimizer_params['lr'], weight_decay=optimizer_params['wd'])
        self.max_grad_norm = optimizer_params['max_grad_norm']
        self.mse_loss = nn.MSELoss()

    def take_action(self, state, valid_actions_mask=None, deterministic=False):
        """
        根据当前策略选择动作。

        Args:
            state (numpy.ndarray): 当前状态。

        Returns:
            tuple: 包含所选动作的索引、动作的对数概率和状态值的元组。
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.action_device)
            action_logits, state_value = self.net(state_tensor)

            if valid_actions_mask is not None:
                mask_tensor = torch.tensor(valid_actions_mask, dtype=torch.bool).to(self.action_device)
            else:
                mask_tensor = (state_tensor[0, 0, :, :].flatten() + state_tensor[0, 1, :, :].flatten() == 0)

            action_logits = action_logits.masked_fill(~mask_tensor, float('-inf'))

            if deterministic:
                action = torch.argmax(action_logits, dim=1).squeeze(0)
                action_log_prob = None
                state_value = None
            else:
                dist = Categorical(logits=action_logits)
                action = dist.sample().squeeze(0)
                action_log_prob = dist.log_prob(action).item()
                state_value = state_value.squeeze().item()

        return action.item(), action_log_prob, state_value

    def update(self, rollout_buffer: RolloutBuffer):
        """
        使用 rollout buffer 中的数据更新策略和价值网络。

        Args:
            rollout_buffer (RolloutBuffer): 包含经验数据的 rollout buffer。

        Returns:
            dict: 包含平均策略损失、平均价值损失和平均熵的元组。如果 buffer 为空，则返回 (None, None, None)。
        """
        self.net = self.net.to(self.update_device)
        states = torch.tensor(rollout_buffer.states[:rollout_buffer.ptr], dtype=torch.float32).to(
            self.update_device)  # 形状: (ptr, C, H, W)
        actions = torch.tensor(rollout_buffer.actions[:rollout_buffer.ptr], dtype=torch.long).to(
            self.update_device)  # 形状: (ptr,)
        # b_old_log_probs 必须是在【数据收集时、掩蔽后】计算得到的对数概率
        old_log_probs = torch.tensor(rollout_buffer.log_probs[:rollout_buffer.ptr], dtype=torch.float32).to(
            self.update_device)  # 形状: (ptr,)
        returns = torch.tensor(rollout_buffer.returns[:rollout_buffer.ptr], dtype=torch.float32).to(
            self.update_device)  # 形状: (ptr,)
        advantages = torch.tensor(rollout_buffer.advantages[:rollout_buffer.ptr], dtype=torch.float32).to(
            self.update_device)  # 形状: (ptr,)

        if rollout_buffer.ptr == 0:
            print("Buffer is Empty! Skipping update.")
            return None, None, None

        batch_size = max(1, rollout_buffer.ptr // self.num_mini_batches)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        grad_norm = None

        for _ in range(self.ppo_epochs):
            indices = np.arange(rollout_buffer.ptr)
            np.random.shuffle(indices)

            for start_idx in range(0, rollout_buffer.ptr, batch_size):
                end_idx = min(start_idx + batch_size, rollout_buffer.ptr)
                batch_indices = indices[start_idx : end_idx]

                if len(batch_indices) == 0:
                    continue

                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_returns = returns[batch_indices]
                b_advantages = advantages[batch_indices]

                # 获得有效动作的掩码
                b_valid_masks = (b_states[:, 0, :, :] + b_states[:, 1, :, :] == 0).flatten(start_dim=1)
                b_invalid_masks = ~b_valid_masks

                b_action_logits, b_values = self.net(b_states)
                b_values = b_values.squeeze(-1)

                b_masked_action_logits = b_action_logits.masked_fill(b_invalid_masks, float('-inf'))
                dist = Categorical(logits=b_masked_action_logits)
                b_new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # 计算策略损失
                ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 计算价值损失
                value_loss = self.mse_loss(b_values, b_returns)

                # 总损失
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # 反向传播与优化
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        if num_updates == 0:
            return None

        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates

        self.net = self.net.to(self.action_device)

        return {
            'policy_loss': -avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'grad_norm': grad_norm.item(),
            'total_loss': self.value_loss_coef * avg_value_loss + avg_policy_loss - self.entropy_coef * avg_entropy
        }

    def save_model(self, file_path):
        """
        保存模型的状态字典。

        Args:
            file_path (str): 保存模型的文件路径。
        """
        torch.save(self.net.state_dict(), file_path)
        print(f"模型已保存到 {file_path}")

    def load_model(self, file_path, map_location=torch.device('cpu')):
        """
        加载模型的状态字典。

        Args:
            file_path (str): 模型的文件路径。
            :param map_location:
        """
        self.net.load_state_dict(torch.load(file_path, map_location=map_location))
        print(f"模型已从 {file_path} 加载")


