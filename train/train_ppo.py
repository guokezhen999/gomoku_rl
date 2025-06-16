import numpy as np
import torch
import wandb

from agents.basic import RandomAgent
from agents.ppo import PPOAgent
from agents.buffer import RolloutBuffer
from game.gomoku import Gomoku

class PPOTrainer:
    def __init__(self, env: Gomoku, agent1: PPOAgent, agent2, buffer: RolloutBuffer, num_episodes,
                 steps_before_update, agent1_plays_first_chance=0.5):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        self.buffer = buffer
        self.num_episodes = num_episodes
        self.max_steps_per_episode = env.board_size * env.board_size
        self.steps_before_update = steps_before_update  # PPO 更新的步数阈值

        self.agent1_plays_first_chance = agent1_plays_first_chance
        self.agent1_id = 1
        self.agent2_id = -1

        self.board_size = env.board_size
        self.num_channels = buffer.num_channels

        self.total_steps_collected = 0

    def _get_valid_actions_mask(self, board_state):
        return board_state.flatten() == 0

    def process_state(self, board_state, current_player_id, last_move_loc):
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
        first_player = self.agent1_id

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
                break

            action = -1
            log_prob = None
            value = None

            # Agent1 的回合
            if current_player_id == self.agent1_id:
                processed_obs_agent1 = self.process_state(obs, self.agent1_id, last_move_loc)
                action, log_prob, value = self.agent1.take_action(processed_obs_agent1,
                                                                  valid_actions_mask, deterministic=False)
            else:
                if isinstance(self.agent2, PPOAgent):
                    processed_obs_agent2 = self.process_state(obs, self.agent2_id, last_move_loc)
                    with torch.no_grad():
                        action, _, _ = self.agent2.take_action(processed_obs_agent2, valid_actions_mask,
                                                               deterministic=False)
                else:
                    action = self.agent2.take_action(obs, valid_actions_mask)

            if action == -1:
                print(f"警告: 未为玩家 {current_player_id} 选择动作，回合提前结束。")
                break

            next_obs, reward_form_env, terminated, _, next_info = self.env.step(action)

            if current_player_id == self.agent1_id:
                self.buffer.store(
                    processed_obs_agent1, action, log_prob, reward_form_env, value, terminated
                )
                episode_reward += reward_form_env
                self.total_steps_collected += 1

            obs = next_obs
            info = next_info
            current_episode_steps += 1

            if terminated:
                break

        # 回合结束后，计算优势估计
        last_val_agent1 = 0.0
        if not terminated:
            # Agent1完成最后一步
            if info['current_player'] == self.agent2_id:
                processed_obs_agent1 = self.process_state(obs, self.agent1_id, info.get('last_move_location'))
                with torch.no_grad():
                    _, _, last_val_agent1 = self.agent1.take_action(processed_obs_agent1,
                                                                    self._get_valid_actions_mask(obs))
        # 如果Agent1在本回合收集到数据
        if self.buffer.ptr > self.buffer.path_start_idx:
            self.buffer.compute_advantages(last_val_agent1, terminated)

        winner = info.get('winner')
        win_stats = "Draw"
        if winner == self.agent1_id:
            win_stats = f"Agent1 (PPO) Wins (玩家 {self.agent1_id})"
        elif winner == self.agent2_id:
            win_stats = f"Agent2 Wins (玩家 {self.agent2_id})"
        print(
            f"回合 {episode_num + 1}/{self.num_episodes} | 步数: {current_episode_steps} | 结果: {win_stats} | "
            f"Agent1 奖励: {episode_reward:.2f} | 缓冲区: {self.buffer.ptr}/{self.buffer.buffer_size}")

        # 检查是否达到更新PPO的条件
        if self.total_steps_collected >= self.steps_before_update and self.buffer.ptr > 0:
            print(f"\n已收集 {self.total_steps_collected} 步数据。正在更新 PPO 智能体...")
            update_info = self.agent1.update(self.buffer)
            if update_info['policy_loss'] is not None:  # 检查更新是否成功进行
                print(f"PPO 更新日志: 策略损失={update_info['policy_loss']:.4f}, 价值损失={update_info['value_loss']:.4f}, "
                      f"熵={update_info['entropy']:.4f}")
            self.buffer.clear()  # 清空缓冲区，开始下一轮数据收集
            self.total_steps_collected = 0  # 重置步数计数器
            print("PPO 更新后缓冲区已清空。\n")
            return update_info, episode_reward
        return None, episode_reward

    def train(self, wandb_log_flag=False, config=None, name=None, model_name=None):
        if wandb_log_flag:
            wandb.init(project="gomoku_rl", config=config, name=name)
        last_update_episode = 0 # 上次更新回合数
        update_step_num = 0 # 总更新次数
        update_rewards = 0.0 # 一轮更新中总回报
        max_win_rate = 0.0
        for episode_num in range(self.num_episodes):
            if np.random.rand() < self.agent1_plays_first_chance:
                agent1_id = 1
            else:
                agent1_id = -1

            self.agent1_id = agent1_id
            self.agent2_id = -agent1_id

            update_info, reward = self.play_episode(episode_num)
            update_rewards += reward


            if update_info is not None and wandb_log_flag:
                update_step_num += 1
                update_episode_num = episode_num - last_update_episode
                win_rate = update_rewards / update_episode_num
                if win_rate > max_win_rate:
                    max_win_rate = win_rate
                    if model_name is not None:
                        self.agent1.save_model(f"{model_name}_max_win_rate.pth")
                last_update_episode = episode_num
                # wandb记录日志
                wandb_logs = update_info
                wandb_logs['episode_nums'] = update_episode_num
                wandb_logs['win_rate'] = win_rate
                wandb_logs['total_episodes'] = episode_num
                wandb.log(
                    wandb_logs, step=update_step_num
                )
                update_rewards = 0

        self.env.close()
        print("训练完成")

if __name__ == "__main__":
    board_size = 15
    action_device = torch.device('cpu')
    update_device = torch.device('mps')
    env = Gomoku()

    ppo_config = {
        "gamma": 0.99,  # 折扣因子
        "gae_lambda": 0.95,  # GAE Lambda 参数
        "clip_epsilon": 0.2,  # PPO 裁剪参数
        "ppo_epochs": 10,  # 每次收集数据后，对数据进行训练的 epoch 数
        "num_mini_batches": 4,  # 将数据分成多少个 mini-batch 进行训练
        "entropy_coef": 0.01,  # 熵奖励系数
        "value_loss_coef": 0.5,  # 价值损失系数
        "opponent": "random",
    }

    optimizer_config = {
        'lr': 1e-5,
        'wd': 5e-4,
        'max_grad_norm': 0.5
    }

    agent1_ppo = PPOAgent(
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
    agent1_ppo.load_model("../models/ppo_0.0.4.pth")

    agent2_ppo = PPOAgent(
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
    agent2_ppo.load_model("../models/ppo_0.0.3.pth")

    buffer_capacity = 1024  # 缓冲区容量，最大可存储步数
    rollout_buffer = RolloutBuffer(
        buffer_size=buffer_capacity,
        board_size=board_size,
        gae_lambda=ppo_config["gae_lambda"],  # 使用与 PPO Agent 相同的 GAE lambda 和 gamma
        gamma=ppo_config["gamma"],
        device=update_device  # Buffer 存储 NumPy 数据，get_batch 会转换为 Tensor 并移到设备
    )

    trainer = PPOTrainer(env=env, agent1=agent1_ppo, agent2=agent2_ppo, buffer=rollout_buffer, num_episodes=5000,
                         steps_before_update=512, agent1_plays_first_chance=0.5)

    trainer.train(wandb_log_flag=False, config=ppo_config, name="ppo_policy_value_net_0.0.4_iter_2_episode_5000",
                  model_name="../models/ppo_0.0.4")
    agent1_ppo.save_model('../models/ppo_0.0.4.pth')







