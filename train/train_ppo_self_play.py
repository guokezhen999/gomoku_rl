import torch
import numpy as np
import wandb
from tqdm import tqdm

from agents.buffer import RolloutBuffer
from agents.ppo import PPOAgent
from game.gomoku import Gomoku


class PPOSelfPlayTrainer:
    def __init__(self, env: Gomoku, agent: PPOAgent, buffer: RolloutBuffer, num_updates: int,
                 steps_per_update: int, agent_plays_first_chance=0.5, model_save_path=None,
                 save_frequency = None, print_log=True):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.num_updates = num_updates
        self.max_steps_per_episode = env.board_size * env.board_size
        self.steps_per_update = steps_per_update  # PPO 更新的步数阈值
        self.print_log = print_log

        self.agent_plays_first_chance = agent_plays_first_chance # Agent 在当前回合扮演玩家 1 (黑方) 的概率
        self.save_frequency = save_frequency # 保存的更新数量

        self.agent_player_id = 1
        self.opponent_player_id = -1

        self.board_size = env.board_size
        self.num_channels = buffer.num_channels

        self.total_steps_collected = 0 # 当前更新周期已收集的总步数
        self.total_episodes_played = 0 # 训练开始以来的总回合数

        # --- 跟踪每轮更新的指标 ---
        self.total_steps = 0
        self.episodes_in_update = 0  # 当前更新周期内对弈的回合数
        self.wins_in_update = 0  # 当前更新周期内 Agent 的胜局数 (以 Agent 扮演的玩家 ID 计算)
        self.losses_in_update = 0  # 当前更新周期内 Agent 的负局数
        self.draws_in_update = 0  # 当前更新周期内平局数
        self.model_save_path = model_save_path  # 保存最佳模型的路径前缀

    def _get_valid_actions_mask(self, board_state):
        # 确保只在空位置落子
        return board_state.flatten() == 0

    def process_state(self, board_state, current_player_id, last_move_loc):
        # --- 此函数似乎已经正确处理了玩家透视 ---
        processed_state = np.zeros((self.num_channels, self.board_size, self.board_size), dtype=np.float32)
        # channel 0: 当前我方棋子 (Current player's stones)
        processed_state[0] = (board_state == current_player_id).astype(np.float32)
        # channel 1: 当前对方棋子 (Opponent's stones)
        processed_state[1] = (board_state == -current_player_id).astype(np.float32)
        # channel 2: 对方最后落子 (Opponent's last move) - 相对于当前玩家是正确的
        if last_move_loc:
            r, c = last_move_loc
            if 0 <= r < self.board_size and 0 <= c < self.board_size:
                # 确保 last_move_loc 确实是对手的落子
                processed_state[2, r, c] = 1.0
        # channel 3: 玩家 ID 特征 (Player ID feature)
        if current_player_id == 1:
            processed_state[3] = np.full((self.board_size, self.board_size), 1.0, dtype=np.float32)
        else:
            processed_state[3] = np.full((self.board_size, self.board_size), 0.0, dtype=np.float32)

        return processed_state

    def play_episode(self):
        if np.random.rand() < self.agent_plays_first_chance:
            self.agent_player_id = 1
            self.opponent_player_id = -1
            first_player = self.agent_player_id  # Agent 先手
        else:
            self.agent_player_id = -1
            self.opponent_player_id = 1
            first_player = self.opponent_player_id  # 对手 (Agent 的策略) 先手

        obs, info = self.env.reset(first_player=first_player)
        terminated = False
        current_episode_steps = 0
        episode_reward = 0

        for step in range(self.max_steps_per_episode):
            current_player_id = info['current_player']
            last_move_loc = info.get('last_move_location')
            valid_actions_mask = self._get_valid_actions_mask(obs)

            if not np.any(valid_actions_mask):
                terminated = True
                info['winner'] = 0
                break

            processed_obs = self.process_state(obs, current_player_id, last_move_loc)

            action = -1
            log_prob = None  # 只在 Agent 回合时记录
            value = None

            if current_player_id == self.opponent_player_id:
                with torch.no_grad():
                    action, _, _ = self.agent.take_action(
                        processed_obs, valid_actions_mask, deterministic=False
                    )
            else:
                action, log_prob, value = self.agent.take_action(
                    processed_obs, valid_actions_mask, deterministic=False
                )

            next_obs, reward, terminated, _, next_info = self.env.step(action)

            if current_player_id == self.agent_player_id:
                self.buffer.store(
                    processed_obs, action, log_prob, reward, value, terminated
                )
                episode_reward += reward
                self.total_steps_collected += 1

            obs = next_obs
            info = next_info
            current_episode_steps += 1

            if terminated:
                break

        # 回合结束处理
        self.total_episodes_played += 1  # 总回合数增加
        self.episodes_in_update += 1

        # 记录当前回合的胜负结果 (从 Agent 的角度)
        winner = info.get('winner')
        if winner == self.agent_player_id:
            self.wins_in_update += 1
            win_stats = f"Agent (玩家 {self.agent_player_id}) 获胜"
        elif winner == self.opponent_player_id:
            self.losses_in_update += 1
            win_stats = f"Opponent (玩家 {self.opponent_player_id}) 获胜"
        else:  # 平局或达到最大步数
            self.draws_in_update += 1
            win_stats = "平局 / 最大步数"

        self.total_steps += current_episode_steps
        if self.print_log:
            print(
                f"回合 {self.total_episodes_played} | 扮演: {self.agent_player_id} | "
                f"步数: {current_episode_steps} | 结果: {win_stats} | "
                f"Agent 回合总奖励: {episode_reward:.2f} | "
                f"当前周期步数: {self.total_steps_collected}/{self.steps_per_update}"
            )

        # 检查是否达到更新条件
        if self.total_steps_collected >= self.steps_per_update and self.buffer.ptr > 0:
            if self.print_log:
                print(f"\n已收集 {self.total_steps_collected} 步数据 (来自 {self.episodes_in_update} 回合)。"
                      f"正在计算优势并更新 PPO 智能体...")
            last_val = 0.0
            if not terminated:
                last_obs_processed_for_agent = self.process_state(obs, self.agent_player_id,
                                                                  info.get('last_move_location'))
                with torch.no_grad():
                    # 从 Agent 的价值网络获取 V(s_{t+1})
                    _, _, last_val = self.agent.take_action(
                        last_obs_processed_for_agent, self._get_valid_actions_mask(obs), deterministic=True
                    )

            self.buffer.compute_advantages(last_val, terminated)

            # 执行更新
            update_info = self.agent.update(self.buffer)

            # 更新日志
            if update_info['policy_loss'] is not None:  # 确保更新成功发生
                if self.print_log:
                    print(
                        f"PPO 更新日志: 策略损失={update_info['policy_loss']:.4f}, "
                        f"价值损失={update_info['value_loss']:.4f}, "
                        f"熵={update_info['entropy']:.4f}"
                    )

                ep_count = self.episodes_in_update
                win_rate = self.wins_in_update / ep_count if ep_count > 0 else 0
                loss_rate = self.losses_in_update / ep_count if ep_count > 0 else 0
                draw_rate = self.draws_in_update / ep_count if ep_count > 0 else 0
                update_info['episodes_per_update'] = ep_count
                update_info['win_rate'] = win_rate
                update_info['loss_rate'] = loss_rate
                update_info['draw_rate'] = draw_rate

            self.buffer.clear()  # 清空缓冲区
            self.total_steps_collected = 0  # 重置步数计数
            self.episodes_in_update = 0  # 重置回合数计数
            self.wins_in_update = 0
            self.losses_in_update = 0
            self.draws_in_update = 0
            if self.print_log:
                print("PPO 更新完成，缓冲区已清空，开始收集下一轮数据。\n")
            return update_info

        return None

    def train(self, wandb_log_flag=False, config=None, name=None):
        if wandb_log_flag:
            if self.model_save_path is None and config:
                self.model_save_path = f"model_{config.get('run_name', 'gomoku_selfplay')}"
                # 将重要的训练超参数和设置添加到 wandb config 中
            run_config = config if config else {}
            run_config.update({
                'steps_per_update': self.steps_per_update,
                'agent_plays_first_chance': self.agent_plays_first_chance,
            })
            wandb.init(project="gomoku_rl_selfplay", config=run_config, name=name)
            wandb.watch(self.agent.net, log='all', log_freq=max(1, self.steps_per_update // 100))

        print(f"开始自博弈训练，目标进行 {self.num_updates} 次 PPO 更新..."
              f"每次更新需要约 {self.steps_per_update} 步 Agent 数据。")

        # 训练主循环
        for update_num in tqdm(range(self.num_updates), desc="PPO 更新进度"):
            update_info = None
            # 持续进行对弈，直到收集到足够的步数触发一次更新
            while update_info is None:
                update_info = self.play_episode()

            if update_info is not None and wandb_log_flag:
                wandb_logs = update_info
                wandb_logs['avg_steps_per_episode'] =  self.total_steps // update_info['episodes_per_update']
                self.total_steps = 0
                wandb.log(wandb_logs, step=update_num + 1)
            if self.save_frequency and (update_num + 1) % self.save_frequency == 0:
                self.agent.save_model(f"{self.model_save_path}_update_{update_num + 1}.pth")

        self.env.close()  # 关闭环境
        print(f"训练完成。总共进行了 {self.num_updates} 次更新，玩了 {self.total_episodes_played} 回合。")
        if self.model_save_path:
            # 训练结束后，保存最终模型
            final_path = f"{self.model_save_path}.pth"
            self.agent.save_model(final_path)
            print(f"最终模型已保存到: {final_path}")

        if wandb_log_flag:
            wandb.finish()


if __name__ == '__main__':
    num_updates = 0  # 总共要进行的 PPO 更新次数 (例如 10000 次更新)
    steps_per_update = 4096

    board_size = 15
    env = Gomoku(board_size=board_size, render_mode=None) # 请确保您的Env返回的状态和信息格式与代码兼容
    action_device = torch.device('cpu')
    update_device = torch.device('mps')

    ppo_config = {
        "gamma": 0.99,  # 折扣因子
        "gae_lambda": 0.95,  # GAE Lambda 参数
        "clip_epsilon": 0.2,  # PPO 裁剪参数
        "ppo_epochs": 20,  # 每次收集数据后，对数据进行训练的 epoch 数
        "num_mini_batches": steps_per_update // 128,  # 将数据分成多少个 mini-batch 进行训练
        "entropy_coef": 0.05,  # 熵奖励系数
        "value_loss_coef": 0.5,  # 价值损失系数
        "opponent": "random"
    }

    optimizer_config = {
        'lr': 2e-4,
        'wd': 5e-4,
        'max_grad_norm': 0.5
    }

    agent = PPOAgent(
        board_size=board_size,
        gamma=ppo_config["gamma"],
        gae_lambda=ppo_config["gae_lambda"],
        clip_epsilon=ppo_config["clip_epsilon"],
        ppo_epochs=ppo_config["ppo_epochs"],
        num_mini_batches=ppo_config["num_mini_batches"],
        entropy_coef=ppo_config["entropy_coef"],
        value_loss_coef=ppo_config["value_loss_coef"],
        action_device=action_device,
        update_device=update_device,
        optimizer_params=optimizer_config
    )
    agent.load_model('../models/ppo_selfplay_0.0.2')

    buffer = RolloutBuffer(
        buffer_size=steps_per_update + 512,
        board_size=board_size,
        gae_lambda=ppo_config["gae_lambda"],
        gamma=ppo_config["gamma"],
        device=update_device
    )

    trainer = PPOSelfPlayTrainer(
        env=env,
        agent=agent,  # 传入您的 PPOAgent 实例
        buffer=buffer,  # 传入您的 RolloutBuffer 实例
        num_updates=num_updates,
        steps_per_update=steps_per_update,
        agent_plays_first_chance=0.5,  # Agent 扮演先手 (黑方) 的概率设为 50%
        model_save_path='../models/ppo_selfplay_0.0.2',
        save_frequency=num_updates // 2,
        print_log=False
    )

    wandb_config = {  # 您的 wandb 配置字典
        "entropy_coef": agent.entropy_coef,
        "value_loss_coef": agent.value_loss_coef,
        "batch_size_ppo": steps_per_update / agent.num_mini_batches,  # PPO 更新的 mini-batch 大小
        "steps_per_update": steps_per_update,  # 训练器参数
        "num_updates": num_updates,  # 训练器参数
        "run_name": "selfplay_ppo_0.0.2"  # wandb 运行名称
    }

    trainer.train(wandb_log_flag=False, config=wandb_config, name=wandb_config["run_name"])











