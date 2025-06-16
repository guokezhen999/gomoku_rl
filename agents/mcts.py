import collections
import math
import time

import torch
import torch.nn.functional as F
import numpy as np
from torch import optim

from agents.basic import Agent
from agents.net import PolicyValueNet


class MCTSNode:
    """MCTS树中的节点。"""
    def __init__(self, parent, prior_p, board_state, current_player, last_move_loc):
        self.parent: MCTSNode = parent # 父节点
        self.children: dict[int, MCTSNode] = {} # 动作 -> 子节点 的映射

        self.board_state = board_state.copy() if board_state is not None else None# 存储该节点的棋盘状态
        self.current_player = current_player # 该状态下轮到谁行动
        self.last_move_loc = last_move_loc # 导致到达此状态的上一步落子的位置 (行, 列)

        self._visit_count = 0 # N(s,a): 从父节点选择动作 a 到达此节点的访问次数
        self._total_value = 0.0 # W(s,a): 通过此节点反向传播的总价值
        self._quality_value = 0.0  # Q(s,a): 平均价值 (W/N)
        self._prior_p = prior_p  # P(s,a): 从父节点选择动作 a 的先验概率

    def expand(self, action_priors):
        """
            扩展当前节点，为所有合法动作创建子节点。
            action_priors: 一个包含 (action, prior_p) 元组的列表。
        """
        for action, prior_p in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior_p=prior_p,
                                                 board_state=None, current_player=None, last_move_loc=None)

    def select_child(self, c_puct):
        """
            使用 PUCT 公式选择要探索的子节点。
            PUCT(s,a) = Q(s,a) + U(s,a)
            U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            其中 N(s) 是父节点(当前节点)的总访问次数。
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None

        parent_visit_count = self.get_visit_count()

        for action, child in self.children.items():
            u_score = (c_puct * child.get_prior_p() *
                       math.sqrt(parent_visit_count) / (1 + child.get_visit_count()))
            score = child.get_quality_value() + u_score
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def update(self, value):
        self._visit_count += 1
        self._total_value += value
        self._quality_value = self._total_value / self._visit_count if self._visit_count > 0 else 0.0

    def update_recursive(self, value):
        """递归地向上更新父节点。"""
        if self.parent:
            self.parent.update_recursive(-value)
        self.update(value)

    def get_visit_count(self):
        return self._visit_count

    def get_quality_value(self):
        return self._quality_value

    def get_prior_p(self):
        return self._prior_p

    def is_leaf(self):
        """检查节点是否为叶子节点（未扩展）。"""
        return len(self.children) == 0

    def __str__(self):
        if isinstance(self._total_value, torch.Tensor):
            total_value = self._total_value.item()
        else:
            total_value = self._total_value
        if isinstance(self._quality_value, torch.Tensor):
            quality_value = self._quality_value.item()
        else:
            quality_value = self._quality_value
        if isinstance(self._prior_p, torch.Tensor):
            prior_p = self._prior_p.item()
        else:
            prior_p = self._prior_p
        return (f"Node(Player:{self.current_player}, N:{self._visit_count}, W:{total_value:.2f}, "
                f"Q:{quality_value:.2f}, P:{prior_p:.3f})")

class MCTS:
    """蒙特卡洛树搜索实现。"""
    def __init__(self, net: PolicyValueNet, c_puct=1.0, n_simulations=100, device=torch.device("cpu"),
                 win_len=5, action_to_location_map=None):
        self.net = net
        self.c_puct = c_puct  # PUCT 公式中的探索常数
        self.n_simulations = n_simulations  # 每次移动执行的 MCTS 模拟次数
        self.device = device

        self.board_size = net.board_size
        self.action_dim = net.board_size * net.board_size
        self.num_channels = 4

        self.win_len = win_len
        self.action_to_location = action_to_location_map
        if self.action_to_location is None:
            raise ValueError("MCTS 初始化需要传入 action_to_location_map")

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

    def get_policy_value(self, node: MCTSNode):
        """使用网络评估节点状态，获取策略和价值。"""
        if node.board_state is None:
            raise ValueError("Node board_state is None, cannot evaluate.")

        state_input = self._preprocess_state(node.board_state, node.current_player, node.last_move_loc)
        state_tensor = torch.tensor(state_input, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.net.eval()
        with torch.no_grad():
            action_logits, value = self.net(state_tensor)

        mask_tensor = (state_tensor[0, 0, :, :].flatten() + state_tensor[0, 1, :, :].flatten() == 0)
        mask_np = mask_tensor.cpu().numpy()
        if not mask_tensor.any():  # 如果没有合法动作
            return [], value
        else:
            action_logits = action_logits.masked_fill(~mask_tensor, float('-inf'))
            action_probs = F.softmax(action_logits, dim=-1)
            action_probs_np = action_probs.cpu().numpy()[0]

        valid_actions = np.where(mask_np)[0]
        valid_action_priors = []

        if len(valid_actions) > 0:
            for action_idx in valid_actions:
                valid_action_priors.append((action_idx, action_probs_np[action_idx]))

        return valid_action_priors, value

    def _simulate_step(self, root_node: MCTSNode):
        # 执行一步 MCTS 模拟：选择、扩展、评估、反向传播。
        current_node = root_node

        # 1. 选择 (Selection) - 沿着树向下走到叶子节点
        while not current_node.is_leaf():
            action, selected_child_node = current_node.select_child(self.c_puct)

            # 如果子节点是第一次访问 (状态未设置)
            if selected_child_node.board_state is None:
                parent_board = current_node.board_state
                player_who_moved = current_node.current_player
                r, c = self.action_to_location[action]

                child_board = parent_board.copy()
                child_board[r, c] = player_who_moved

                # 设置子节点的状态、下一个玩家以及导致此状态的最后一步位置
                selected_child_node.board_state = child_board
                selected_child_node.current_player = -player_who_moved  # 轮到对手
                selected_child_node.last_move_loc = (r, c)  # 动作 (r,c)导致了当前子节点状态

            # 移动到选中的子节点，继续向下搜索
            current_node = selected_child_node

        is_terminal, winner = self._check_terminal(current_node.board_state, current_node.last_move_loc)

        if is_terminal:
            if winner == 0:
                value = 0.0
            else:
                value = 1.0 if winner == current_node.current_player else -1.0
            current_node.update_recursive(value)
            return

        # 2. 扩展 (Expansion) & 3. 评估 (Evaluation)
        valid_action_priors, value_estimate = self.get_policy_value(current_node)
        # 如果有合法动作，则扩展节点
        if valid_action_priors:
            current_node.expand(valid_action_priors)

        # 4. 反向传播 (Backpropagation)
        current_node.update_recursive(value_estimate)

    def run_simulations(self, root_node: MCTSNode):
        for _ in range(self.n_simulations):
            self._simulate_step(root_node)


    def _check_terminal(self, board_state, last_move_loc):
        if last_move_loc is not None:
            r, c = last_move_loc
            player_moved = board_state[r, c]
            if player_moved != 0:
                if self._check_win_at(board_state, player_moved, r, c):
                    return True, player_moved

        if np.all(board_state != 0):
            return True, 0

        return False, None

    def _check_win_at(self, board_state, player_moved, r, c):
        win_len = self.win_len
        # 检查四个方向
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # 向一个方向检查
            for i in range(1, win_len):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board_state[nr, nc] == player_moved:
                    count += 1
                else:
                    break
            # 向相反方向检查
            for i in range(1, win_len):
                nr, nc = r - i * dr, c - i * dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board_state[nr, nc] == player_moved:
                    count += 1
                else:
                    break
            # 如果连子数达到 win_len，返回 True
            if count >= win_len: return True
        return False

    def get_policy(self, node: MCTSNode, temperature=1.0):
        # 根据子节点的访问次数计算策略概率分布
        # temperature 控制探索程度：
        #   temp=1.0: 概率正比于访问次数。
        #   temp->0: 趋向于选择访问次数最多的动作 (贪婪)。
        #   temp->inf: 趋向于均匀选择。
        # 返回一个大小为 action_dim 的向量，包含每个动作的 MCTS 概率。
        visit_counts = np.array([child.get_visit_count() for child in node.children.values()])
        actions = [action for action in node.children.keys()]

        if not actions:
            return np.zeros(self.action_dim, dtype=np.int32)

        if temperature == 0: # 贪婪选择访问次数最多的动作
            best_action_idx = np.argmax(visit_counts)
            policy = np.zeros_like(visit_counts, dtype=float)
            policy[best_action_idx] = 1.0
        else:
            visit_powers = np.power(visit_counts, 1.0 / temperature)
            policy_sum = np.sum(visit_powers)
            if policy_sum > 1e-6:
                policy = visit_powers / policy_sum
            else:
                policy = np.ones_like(visit_counts) / len(visit_counts)

        # 构建完整的策略向量 (大小为 action_dim)，将合法动作的概率填入对应位置
        full_policy = np.zeros(self.action_dim, dtype=np.float32)
        for action, prob in zip(actions, policy):
            full_policy[action] = prob

        return full_policy

class MCTSAgent(Agent):
    def __init__(self, net: PolicyValueNet, n_simulations=100, c_puct=1.0, temperature=1.0,
                 device=torch.device("cpu"), win_len=5, action_to_location_map=None):
        super().__init__(net.action_dim)
        self.net = net
        self.mcts = MCTS(net, c_puct, n_simulations, device, win_len, action_to_location_map)
        self.n_simulations = n_simulations
        self.temperature = temperature

    def take_action(self, board_state,  current_player, last_move_loc,
                      add_dirichlet_noise=False, dirichlet_alpha=0.3, noise_epsilon=0.25):
        # 1. 创建 MCTS 树的根节点
        root_node = MCTSNode(parent=None, prior_p=1.0, board_state=board_state,
                             current_player=current_player, last_move_loc=last_move_loc)

        # 2. 使用网络预测根节点的策略先验和价值
        valid_action_priors, _ = self.mcts.get_policy_value(root_node)

        if not valid_action_priors:
            return -1, np.zeros(self.action_dim, dtype=np.float32)

        # 3. (可选) 添加 Dirichlet noise 到根节点的先验概率以促进探索 (用于训练数据生成)
        if add_dirichlet_noise:
            actions = [a for a, _ in valid_action_priors]
            priors = np.array([p for _, p in valid_action_priors])
            # 生成 Dirichlet noise，与合法动作数量相同
            noise = np.random.dirichlet([dirichlet_alpha] * len(priors))
            noisy_priors = (1 - noise_epsilon) * priors + noise_epsilon * noise
            final_priors = list(zip(actions, noisy_priors))
            root_node.expand(final_priors)
        else:
            root_node.expand(valid_action_priors)

        # 4. 运行 MCTS 模拟
        self.mcts.run_simulations(root_node)

        # 5. 获取 MCTS 生成的策略 (基于访问次数)
        # mcts_policy 是一个大小为 action_dim 的向量
        mcts_policy = self.mcts.get_policy(root_node, temperature=self.temperature)

        # 6. 根据 MCTS 策略选择动作
        valid_actions = np.where(mcts_policy > 0)[0]

        if len(valid_actions) == 0:
            print("警告: MCTS 策略全为零。回退到网络直接预测的合法动作。")
            if valid_action_priors:
                # Fallback: 如果 MCTS policy 是空的，从网络直接输出的 legal_action_priors 中采样
                actions = [a for a, _ in valid_action_priors]
                probs = np.array([p for _, p in valid_action_priors])
                # 再次归一化以确保概率和为 1
                if np.sum(probs) < 1e-8:
                    probs = np.ones_like(probs) / len(probs)
                else:
                    probs = probs / np.sum(probs)

                if len(actions) > 0:
                    action = np.random.choice(actions, p=probs)
                else:  # 网络也没有合法动作
                    return -1, mcts_policy  # 返回无动作
            else:  # 网络也没有合法动作
                return -1, mcts_policy  # 返回无动作
        else:
            actual_probs = mcts_policy[valid_actions]
            action = np.random.choice(valid_actions, p=actual_probs / np.sum(actual_probs))

        return action, mcts_policy

    def load_model(self, file_path, map_location=torch.device('cpu')):
        """
        加载模型的状态字典。

        Args:
            file_path (str): 模型的文件路径。
            :param map_location:
        """
        self.net.load_state_dict(torch.load(file_path, map_location=map_location))
        print(f"模型已从 {file_path} 加载")











        







