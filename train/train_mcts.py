import collections
import time

import numpy as np
import torch
import wandb
from torch import optim
from torch.nn import functional as F

from agents.mcts import MCTSAgent
from agents.net import PolicyValueNet
from game.gomoku import Gomoku


class AlphaZeroTrainer:
    def __init__(self, env: Gomoku, net: PolicyValueNet,
                 num_episodes=1000,  # 总共生成和训练的回合数 (自我对弈次数)
                 mcts_simulations=50,  # 每个动作的 MCTS 模拟次数
                 learning_rate=1e-3,  # 学习率
                 weight_decay=5e-4,  # 正则系数
                 batch_size=64,  # 训练时的 batch 大小
                 buffer_size=10000,  # 存储 (s, pi, z) 数据的缓冲区最大容量
                 epochs_per_update=4,  # 每次模型更新时，对缓冲区数据训练的轮数
                 update_threshold=500,  # 缓冲区数据量达到多少时开始训练
                 c_puct=1.0,  # MCTS PUCT 公式中的探索常数
                 temperature_start=1.0,  # 自我对弈开始时的采样温度
                 temperature_decay=0.97,  # 采样温度衰减率 (每隔一定回合衰减)
                 min_temperature=0.1,  # 采样温度最低值
                 dirichlet_alpha=0.3,  # Dirichlet noise 参数 alpha
                 noise_epsilon=0.25,  # Dirichlet noise 混合权重
                 checkpoint_freq=100,  # 每隔多少回合保存一次模型检查点
                 action_device=torch.device("cpu"),
                 update_device=torch.device("mps")):  # 使用的计算设备


        self.env = env
        self.net = net.to(action_device)

        # 初始化 MCTSAgent，传入网络和 MCTS 参数
        self.agent = MCTSAgent(net, n_simulations=mcts_simulations, c_puct=c_puct,
                               device=action_device, win_len=env.win_len,  # 从环境获取获胜长度
                               action_to_location_map=env.action_to_location)
        self.num_episodes = num_episodes
        self.batch_size = batch_size

        self.buffer = collections.deque(maxlen=buffer_size)
        self.epochs_per_update = epochs_per_update
        self.update_threshold = update_threshold
        self.lr = learning_rate
        self.wd = weight_decay
        # 使用 Adam 优化器
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.temperature = temperature_start
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_epsilon = noise_epsilon
        self.checkpoint_freq = checkpoint_freq
        self.action_device = action_device
        self.update_device = update_device
        self.board_size = env.board_size
        self.action_dim = env.action_space.n
        self.num_channels = 4

    def _preprocess_state(self, board_state, current_player_id, last_move_loc):
        processed_state = np.zeros((self.num_channels, self.board_size, self.board_size), dtype=np.float32)
        # channel 0: 当前我方棋子
        processed_state[0] = (board_state == current_player_id).astype(np.float32)
        # channel 1: 当前对方棋子
        processed_state[1] = (board_state == -current_player_id).astype(np.float32)
        # channel 2: 对方最后落子
        if last_move_loc:
            r, c = last_move_loc
            if 0 <= r < self.board_size and 0 <= c < self.board_size:
                processed_state[2, r, c] = 1.0
        # channel 3: 玩家 ID (1.0 对于 id=1, 0.0 对应id=-1)
        if current_player_id == 1:
            processed_state[3] = np.full((self.board_size, self.board_size), 1.0, dtype=np.float32)
        else:
            processed_state[3] = np.full((self.board_size, self.board_size), 0.0, dtype=np.float32)
        return processed_state

    def play_episode(self, episode_num):
        print(f"--- 开始自我对弈回合 {episode_num + 1} ---")
        start_time = time.time()
        episode_data = []

        obs, info = self.env.reset()
        current_player = info['current_player']
        last_move_loc = info['last_move_location']

        terminated = False
        steps = 0

        current_temp = self.temperature * (self.temperature_decay ** (episode_num // 10))
        current_temp = max(current_temp, self.min_temperature)
        self.agent.temperature = current_temp
        print("current temperature = ", current_temp)

        while not terminated:
            processed_state = self._preprocess_state(obs, current_player, last_move_loc)

            add_noise = True  # 可以根据回合数或其他条件来决定是否添加噪声
            action, mcts_policy = self.agent.take_action(
                obs.copy(),  # 传递棋盘状态副本
                current_player,
                last_move_loc,  # 传递最后一步位置给 Agent
                add_dirichlet_noise=add_noise,
                dirichlet_alpha=self.dirichlet_alpha,
                noise_epsilon=self.noise_epsilon
            )

            # 存储当前状态的数据：(NN输入状态, MCTS策略 Pi, 当前玩家)
            episode_data.append((processed_state, mcts_policy, current_player))

            next_obs, reward, terminated, _, info = self.env.step(action)
            last_move_loc = info['last_move_location']

            obs = next_obs
            current_player = info['current_player']
            steps += 1

        # 将本回合收集的数据添加到buffer
        winner = info['winner']
        num_samples = len(episode_data)
        for i in range(num_samples):
            state_input, policy_target, player = episode_data[i]
            if winner is not None:
                value_target = float(winner) if player == 1 else -float(winner)
            else:
                value_target = 0.0
            self.buffer.append((state_input, policy_target, value_target))

        end_time = time.time()
        result_str = "平局"
        episode_return = 0.0
        if winner == 1:
            result_str = "玩家 1 (黑方) 胜利"
            episode_return = 1.0
        elif winner == -1:
            result_str = "玩家 -1 (红方) 胜利"
            episode_return = -1.0
        print(f"回合 {episode_num + 1} 结束，共 {steps} 步。结果: {result_str}. "
              f"耗时: {end_time - start_time:.2f}s. 缓冲区大小: {len(self.buffer)}")
        return steps, episode_return

    def train_step(self):
        print(f"--- 开始训练步骤 --- 缓冲区大小: {len(self.buffer)}")
        total_loss_accum = 0  # 累计总损失
        policy_loss_accum = 0  # 累计策略损失
        value_loss_accum = 0  # 累计价值损失
        num_batches = 0  # 累计批次数量

        avg_total_loss, avg_policy_loss, avg_value_loss = None, None, None

        self.net = self.net.to(self.update_device)
        self.net.train()

        for epoch in range(self.epochs_per_update):
            num_batches_epoch = 0  # 当前 epoch 的批次数量
            buffer_len = len(self.buffer)
            indices = np.arange(buffer_len)
            np.random.shuffle(indices)

            for start_idx in range(0, buffer_len, self.batch_size):
                end_idx = min(start_idx + self.batch_size, buffer_len)
                batch_indices = indices[start_idx: end_idx]
                mini_batch = [self.buffer[i] for i in batch_indices]
                state_batch, policy_target_batch, value_target_batch = zip(*mini_batch)

                state_batch_np = np.array(state_batch)
                policy_target_batch_np = np.array(policy_target_batch)
                value_target_batch_np = np.array(value_target_batch).reshape(-1, 1)
                states_tensor = torch.tensor(state_batch_np, dtype=torch.float32).to(self.update_device)
                policy_targets_tensor = torch.tensor(policy_target_batch_np, dtype=torch.float32).to(self.update_device)
                value_targets_tensor = torch.tensor(value_target_batch_np, dtype=torch.float32).to(self.update_device)

                self.optimizer.zero_grad()
                action_logits_pred, value_pred = self.net(states_tensor)

                # 计算策略损失：交叉熵损失
                policy_loss = F.cross_entropy(action_logits_pred, policy_targets_tensor)

                # 计算价值损失：均方误差损失
                value_loss = F.mse_loss(value_pred, value_targets_tensor)

                # 总损失是策略损失和价值损失的和
                total_loss = policy_loss + value_loss
                # 反向传播计算梯度
                total_loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                # 执行一步优化器，更新网络权重
                self.optimizer.step()

                # 累加损失值，用于计算平均损失
                total_loss_accum += total_loss.item()
                policy_loss_accum += policy_loss.item()
                value_loss_accum += value_loss.item()
                num_batches_epoch += 1  # 当前 epoch 的批次数量加一

            num_batches += num_batches_epoch
            if num_batches > 0:
                avg_total_loss = total_loss_accum / num_batches
                avg_policy_loss = policy_loss_accum / num_batches
                avg_value_loss = value_loss_accum / num_batches
                print(f"训练完成。平均总损失: {avg_total_loss:.4f}, "
                      f"平均策略损失: {avg_policy_loss:.4f}, 平均价值损失: {avg_value_loss:.4f}, 最终梯度: {grad.item():.4f}")

        self.net = self.net.to(self.action_device)
        self.net.eval()
        return avg_total_loss, avg_policy_loss, avg_value_loss

    def train(self, save_path, wandb_log=False, config=None):
        print("--- 开始 AlphaZero 训练循环 ---")
        if wandb_log:
            wandb.init(project="gomoku_rl_mcts", config=config, name=config['name'])
        for i_episode in range(self.num_episodes):
            # 进行一回合自我对弈，收集数据
            self.play_episode(i_episode)

            # 检查是否达到训练条件并进行训练
            # 例如，每隔 1 回合检查一次，并且缓冲区数据量达到阈值
            if len(self.buffer) >= self.update_threshold:
                total_loss, policy_loss, value_loss = self.train_step()
                if wandb_log:
                    wandb.log({
                        "total_loss": total_loss,
                        "policy_loss": policy_loss,
                        "value_loss": value_loss
                    }, step=i_episode + 1)

            # 保存模型检查点
            if (i_episode + 1) % self.checkpoint_freq == 0:
                model_path = f"{save_path}_episode_{i_episode + 1}.pth"
                torch.save(self.net.state_dict(), model_path)

        print("--- 训练循环结束 ---")
        torch.save(self.net.state_dict(), f'{save_path}.pth')

if __name__ == '__main__':
    device = torch.device('mps')

    board_size = 9
    win_len = 5
    num_channels = 4

    env = Gomoku(board_size, win_len)
    net = PolicyValueNet(board_size)

    save_path = '../models/mcts_9.0.2'

    trainer = AlphaZeroTrainer(env=env, net=net, num_episodes=50, mcts_simulations=1000, learning_rate=5e-5,
                               weight_decay=5e-4, batch_size=128, buffer_size=1200, epochs_per_update=40,
                               update_threshold=200, c_puct=1.0, temperature_start=0.8, temperature_decay=0.99,
                               min_temperature=0.1, dirichlet_alpha=0.3, noise_epsilon=0.25, checkpoint_freq=10,
                               action_device=torch.device('cpu'), update_device=torch.device('mps'))

    trainer.agent.load_model('../models/mcts_9.0.2.pth')

    wandb_config = {
        "name": "mcts_9.0.2",
        "num_episodes": trainer.num_episodes,
        "learning_rate": trainer.lr,
        "weight_decay": trainer.wd,
        "batch_size": trainer.batch_size,
        "buffer_size": trainer.buffer.maxlen,
        "epochs_per_update": trainer.epochs_per_update
    }

    trainer.train(save_path, wandb_log=True, config=wandb_config)



