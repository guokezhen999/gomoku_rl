import numpy as np
import torch
import pygame

from agents.basic import Agent, RandomAgent
from agents.mcts import MCTSAgent
from agents.net import PolicyValueNet
from agents.ppo import PPOAgent
from game.gomoku import Gomoku

class GomokuTester:
    def __init__(self, env: Gomoku, test_agent: Agent, opponent_agent: Agent, num_test_episodes,
                 device, test_agent_id=1, test_player_first_chance=0.5):
        self.env = env
        self.test_agent = test_agent
        self.opponent_agent = opponent_agent
        self.num_test_episodes = num_test_episodes
        self.device = device
        self.test_agent_id = test_agent_id
        self.opponent_agent_id = -test_agent_id
        self.test_player_first_chance = test_player_first_chance

        self.board_size = env.board_size
        self.num_channels = 4
        self.max_steps_per_episode = self.board_size * self.board_size

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

    def play_episode(self, episode_num, first_player_id):
        obs, info = self.env.reset(first_player=first_player_id)
        current_episode_steps = 0

        for step in range(self.max_steps_per_episode):
            current_player_id = info['current_player']
            last_move_loc = info.get("last_move_location")
            valid_actions_mask = self._get_valid_actions_mask(obs)

            if not np.any(valid_actions_mask):
                break

            if current_player_id == self.test_agent_id:
                if isinstance(self.test_agent, PPOAgent):
                    processed_obs = self.process_state(obs, current_player_id, last_move_loc)
                    action, _, _ = self.test_agent.take_action(processed_obs, valid_actions_mask, True)
                elif isinstance(self.test_agent, MCTSAgent):
                    action, _ = self.test_agent.take_action(obs, current_player_id, last_move_loc,
                                                0.0, False)
                else:
                    action = self.test_agent.take_action(obs.copy(), valid_actions_mask=valid_actions_mask)
            else:
                if isinstance(self.opponent_agent, PPOAgent):
                    processed_obs = self.process_state(obs, current_player_id, last_move_loc)
                    action, _, _ = self.opponent_agent.take_action(processed_obs, valid_actions_mask, True)
                elif isinstance(self.opponent_agent, MCTSAgent):
                    action, _ = self.opponent_agent.take_action(obs, current_player_id, last_move_loc,
                                                0.0, False)
                else:
                    action = self.opponent_agent.take_action(obs.copy(), valid_actions_mask=valid_actions_mask)


            if action == -1:
                print(f"Warning: No action selected for player {current_player_id}. Episode ending early.")
                break

            next_obs, reward, terminated, _, next_info = self.env.step(action)

            obs = next_obs
            info = next_info
            current_episode_steps += 1

            if terminated:
                break

        return info.get('winner'), current_episode_steps

    def run(self):
        win_as_first, loss_as_first, win_as_second, loss_as_second = 0, 0, 0, 0
        win_count, loss_count, draw_count = 0, 0, 0
        first_episodes, second_episodes = 0, 0
        total_steps = 0

        test_player_id_sequence = np.random.random(self.num_test_episodes)
        test_player_id_sequence = np.where(test_player_id_sequence < self.test_player_first_chance, 1, -1)

        for episode_num in range(self.num_test_episodes):
            # 选取测试玩家序号
            self.test_agent_id = test_player_id_sequence[episode_num]
            self.opponent_agent_id = -self.test_agent_id
            # 选取测试玩家是否先手
            first_player_id = 1
            is_test_agent_first = (first_player_id == self.test_agent_id)
            print(f"测试回合 {episode_num+1}/{self.num_test_episodes} | {self.test_agent.__class__.__name__} "
                  f"({self.test_agent_id}) 先手: {is_test_agent_first}")

            winner, steps = self.play_episode(episode_num, first_player_id=first_player_id)
            total_steps += steps

            if is_test_agent_first:
                first_episodes += 1
            else:
                second_episodes += 1

            if winner == self.test_agent_id:
                win_count += 1
                if is_test_agent_first:
                    win_as_first += 1
                else:
                    win_as_second += 1
                print(f"回合 {episode_num + 1} 结果: {self.test_agent.__class__.__name__} 胜利！")
            elif winner == self.opponent_agent_id:
                loss_count += 1
                print(f"回合 {episode_num+1} 结果: {self.test_agent.__class__.__name__} 失败。")
            else:
                draw_count += 1
                print(f"回合 {episode_num + 1} 结果: 平局。")

            if self.env.render_mode == 'human':
                keep_display_open = True
                while keep_display_open:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            keep_display_open = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                keep_display_open = False

        print("\n---- 测试结束 ----")
        print(f"总回合数: {self.num_test_episodes}")
        print(f"{self.test_agent.__class__.__name__} 表现:")
        print(f"  胜利数: {win_count} ({win_count / self.num_test_episodes:.2%})")
        print(f"  失败数: {loss_count} ({loss_count / self.num_test_episodes:.2%})")
        print(f"  平局数: {draw_count} ({draw_count / self.num_test_episodes:.2%})")
        print(f"  先手胜率: {win_as_first}/{first_episodes} ({win_as_first / (first_episodes + 1e-8):.2%})")
        print(f"  后手胜率: {win_as_second}/{second_episodes} ({win_as_second / (second_episodes + 1e-8):.2%})")
        print(f"平均步数: {total_steps / self.num_test_episodes:.2f}")

        self.env.close()

if __name__ == '__main__':
    board_size = 9
    device = torch.device('cpu')
    env = Gomoku(board_size, render_mode='human')

    # ppo_config = {
    #     "lr": 2e-4,  # 学习率
    #     "gamma": 0.99,  # 折扣因子
    #     "gae_lambda": 0.95,  # GAE Lambda 参数
    #     "clip_epsilon": 0.2,  # PPO 裁剪参数
    #     "ppo_epochs": 10,  # 每次收集数据后，对数据进行训练的 epoch 数
    #     "num_mini_batches": 3,  # 将数据分成多少个 mini-batch 进行训练
    #     "entropy_coef": 0.01,  # 熵奖励系数
    #     "value_loss_coef": 0.5,  # 价值损失系数
    #     "opponent": "random",
    # }
    #
    # optimizer_config = {
    #     'lr': 2e-4,
    #     'wd': 5e-4,
    #     'max_grad_norm': 0.5
    # }
    #
    # agent1_ppo = PPOAgent(
    #     board_size=board_size,
    #     gamma=ppo_config["gamma"],
    #     gae_lambda=ppo_config["gae_lambda"],
    #     clip_epsilon=ppo_config["clip_epsilon"],
    #     ppo_epochs=ppo_config["ppo_epochs"],
    #     num_mini_batches=ppo_config["num_mini_batches"],
    #     entropy_coef=ppo_config["entropy_coef"],
    #     value_loss_coef=ppo_config["value_loss_coef"],
    #     action_device=device,
    #     update_device=device,
    #     optimizer_params=optimizer_config
    # )
    # agent1_ppo.load_model("../models/ppo_selfplay_0.0.1.pth")
    #
    # agent2_ppo = PPOAgent(
    #     board_size=board_size,
    #     gamma=ppo_config["gamma"],
    #     gae_lambda=ppo_config["gae_lambda"],
    #     clip_epsilon=ppo_config["clip_epsilon"],
    #     ppo_epochs=ppo_config["ppo_epochs"],
    #     num_mini_batches=ppo_config["num_mini_batches"],
    #     entropy_coef=ppo_config["entropy_coef"],
    #     value_loss_coef=ppo_config["value_loss_coef"],
    #     action_device=device,
    #     update_device=device,
    #     optimizer_params=optimizer_config
    # )
    # agent2_ppo.load_model("../models/ppo_selfplay_0.0.1.pth")

    net = PolicyValueNet(board_size)
    agent_mcts = MCTSAgent(net, n_simulations=1000, c_puct=1.0, temperature=0, action_to_location_map=env.action_to_location)
    agent_mcts.load_model("../models/mcts_9.0.2.pth")

    agent_random = RandomAgent(board_size)

    tester = GomokuTester(
        env=env,
        test_agent=agent_mcts,
        opponent_agent=agent_random,
        num_test_episodes=100,
        device=device,
    )

    tester.run()









