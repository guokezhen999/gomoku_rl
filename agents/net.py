import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PolicyValueNet(nn.Module):
    """
    策略价值网络，用于评估棋盘状态并预测下一步行动的概率分布。

    Args:
        board_size (int): 棋盘的边长。
    """

    def __init__(self, board_size: int, num_channels=4):
        super().__init__()
        self.board_size = board_size
        self.action_dim = board_size * board_size
        self.num_channels = num_channels
        # 公共卷积层：用于提取棋盘状态的特征。
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)  # 输入4个通道（例如，当前玩家、对手、历史行动等），输出32个通道。
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输出64个通道。
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 输出128个通道。

        # 策略（action）专有层：用于预测下一步行动的概率分布。
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)  # 将128个通道转换为4个通道。
        self.act_fc1 = nn.Linear(4 * self.action_dim, self.action_dim)  # 将卷积层的输出展平，并输入到全连接层，输出动作数量的概率。

        # 价值（value）专有层：用于评估当前棋盘状态的价值。
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)  # 将128个通道转换为2个通道。
        self.val_fc1 = nn.Linear(2 * self.action_dim, 64)  # 将卷积层的输出展平，并输入到全连接层。
        self.val_fc2 = nn.Linear(64, 1)  # 输出一个标量，表示当前状态的价值。

    def forward(self, state):
        """
        前向传播函数。

        Args:
            state (torch.Tensor): 棋盘状态的输入，形状为 (N, 4, board_size, board_size)，
                                  其中 N 是批次大小，4 是通道数。

        Returns:
            tuple: 包含策略（行动概率分布）和价值的元组。
                   - 策略 (torch.Tensor): 形状为 (N, action_dim) 的张量，表示每个行动的对数概率。
                   - 价值 (torch.Tensor): 形状为 (N, 1) 的张量，表示当前状态的价值。
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 策略分支
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.action_dim)  # 展平卷积层的输出。
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)  # 使用 log_softmax 函数，确保输出是概率分布。

        # 价值分支
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.action_dim)  # 展平卷积层的输出。
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))  # 使用 tanh 函数将价值限制在 [-1, 1] 之间。
        return x_act, x_val


