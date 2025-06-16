from gymnasium import spaces
from gymnasium import Env
import numpy as np
import pygame

CELL_SIZE = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

class Gomoku(Env):
    """
    五子棋环境。

    metadata:
        render_modes (list): 支持的渲染模式，包括 "human" (人类观看) 和 "rgb_array" (RGB数组)。
        render_fps (int): 渲染帧率。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, board_size=15, win_len=5, first_hand=1, render_mode=None):
        """
        初始化五子棋环境。

        参数:
            action_dim (int): 棋盘大小。
            win_len (int): 获胜所需的棋子数量。
            first_hand (int): 先手玩家，1代表黑子，-1代表白子。
            render_mode (str, optional): 渲染模式，默认为 None。可以是 "human" 或 "rgb_array"。

        Raises:
            AssertionError: 如果棋盘大小小于获胜所需的棋子数量。
        """
        super().__init__()
        assert board_size >= win_len, "棋盘大小必须大于等于获胜长度"
        self.board_size = board_size
        self.win_len = win_len

        # 观测空间：棋盘状态，-1代表白子，0代表空，1代表黑子
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int32
        )

        # 动作空间：棋盘上的所有位置
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.action_to_location = {i: (i // board_size, i % board_size) for i in range(board_size * board_size)}
        self._location_to_action = {v: k for k, v in self.action_to_location.items()}

        self.current_player = first_hand
        self._first_hand_player = first_hand
        self._last_move_location = None
        self._board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self._winner = None

        self.screen_size = CELL_SIZE * self.board_size
        self.window = None
        self.clock = None
        self.font = None
        self.render_mode = render_mode
        if self.render_mode == "human":
            import pygame
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Gomoku")
            self.clock = pygame.time.Clock()
            try:
                self.font = pygame.font.Font(pygame.font.get_default_font(), 24)
            except pygame.error:
                self.font = pygame.font.SysFont(None, 24)

    def _get_obs(self):
        """
        获取当前观测。

        返回:
            np.ndarray: 当前棋盘状态的副本。
        """
        return self._board.copy()

    def _get_info(self):
        """
        获取当前信息。

        返回:
            dict: 包含当前玩家、最后一步的位置和胜者的信息的字典。
        """
        return {
            "current_player": self.current_player,
            "last_move_location": self._last_move_location,
            "winner": self._winner
        }

    def reset(self, seed=None, first_player=None):
        """
        重置环境。

        参数:
            seed (int, optional): 随机种子，默认为 None。
            first_player (int, optional): 先手玩家，默认为 None。

        返回:
            tuple: 包含观测和信息的元组。
        """
        super().reset(seed=seed)
        self._board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        if first_player:
            self.current_player = first_player
        else:
            self.current_player = self._first_hand_player
        self.current_player = 1 # always start with black
        self._winner = None
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def _is_valid_action(self, row, col):
        """
        检查动作是否有效。

        参数:
            row (int): 行索引。
            col (int): 列索引。

        返回:
            bool: 如果动作有效则返回 True，否则返回 False。
        """
        return self.board_size > row >= 0 == self._board[row, col] and \
            0 <= col < self.board_size

    def _check_win(self, player, r, c):
        """
        检查玩家是否获胜。

        参数:
            player (int): 玩家，1代表黑子，-1代表白子。
            r (int): 行索引。
            c (int): 列索引。

        返回:
            bool: 如果玩家获胜则返回 True，否则返回 False。
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, self.win_len):
                nr, nc  = r + i * dr, c + i * dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self._board[nr, nc] == player:
                    count += 1
                else:
                    break
            for i in range(1, self.win_len):
                nr, nc = r - i * dr, c - i * dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self._board[nr, nc] == player:
                    count += 1
                else:
                    break
            if count >= self.win_len:
                return True
        return False

    def _check_draw(self):
        """
        检查是否平局。

        返回:
            bool: 如果平局则返回 True，否则返回 False。
        """
        return np.all(self._board != 0)

    def step(self, action):
        """
        执行一步动作。

        参数:
            action (int): 动作，棋盘上的位置索引。

        返回:
            tuple: 包含观测、奖励、是否结束和信息的元组。
        """
        if not isinstance(action, (int, np.integer)):
            raise TypeError(f"动作必须是整数, 得到 {type(action)}")
        if not self.action_space.contains(action):
            raise ValueError(f"动作 {action} 对于动作空间无效.")

        row, col = self.action_to_location[action]

        if not self._is_valid_action(row, col):
            return self._get_obs(), -1, False, False, self._get_info()

        self._board[row, col] = self.current_player
        self._last_move_location = (row, col)

        if self.render_mode == "human":
            self._render_frame()

        terminated = self._check_win(self.current_player, row, col)
        if terminated:
            self._winner = self.current_player
            reward = 1.0
            info = self._get_info()
            return self._get_obs(), reward, terminated, False, info
        if self._check_draw():
            self._winner = 0
            reward = 0.0
            terminated = True
            info = self._get_info()
            return self._get_obs(), reward, terminated, False, info

        reward = 0.0
        self.current_player *= -1
        info = self._get_info()
        return self._get_obs(), reward, terminated, False, info

    def render(self):
        """
        渲染环境。

        返回:
            None 或 np.ndarray: 如果渲染模式是 "human" 则返回 None，如果渲染模式是 "rgb_array" 则返回 RGB 数组。
        """
        if self.render_mode in ["human", "rgb_array"]:
            return self._render_frame()
        return None

    def _render_frame(self):
        """
        渲染一帧。
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Gomoku")
            self.font = pygame.font.SysFont('Arial', 20)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface((self.screen_size, self.screen_size))
        canvas.fill(WHITE)

        for i in range(1, self.board_size):
            pygame.draw.line(canvas, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, self.screen_size), 1)
            pygame.draw.line(canvas, BLACK, (0, i * CELL_SIZE), (self.screen_size, i * CELL_SIZE), 1)
        pygame.draw.rect(canvas, BLACK, (0, 0, self.screen_size, self.screen_size), 1)

        radius = CELL_SIZE // 2 - 3
        for r_idx in range(self.board_size):
            for c_idx in range(self.board_size):
                piece_value = self._board[r_idx, c_idx]
                if piece_value != 0:
                    center_x = c_idx * CELL_SIZE + CELL_SIZE // 2
                    center_y = r_idx * CELL_SIZE + CELL_SIZE // 2
                    color = BLACK if piece_value == 1 else RED
                    is_last_move = (self._last_move_location is not None and
                                    self._last_move_location == (r_idx, c_idx))
                    if is_last_move:
                        alpha_value = 128
                        piece_surface_size = (radius * 2 + 4, radius * 2 + 4)
                        piece_surface = pygame.Surface(piece_surface_size,
                                                       pygame.SRCALPHA)
                        piece_surface_center = (piece_surface_size[0] // 2, piece_surface_size[1] // 2)
                        pygame.draw.circle(piece_surface, color, piece_surface_center, radius)
                        piece_surface.set_alpha(alpha_value)
                        blit_x = center_x - piece_surface_center[0]
                        blit_y = center_y - piece_surface_center[1]
                        canvas.blit(piece_surface, (blit_x, blit_y))
                    else:
                        pygame.draw.circle(canvas, color, (center_x, center_y), radius)

        if self._winner is not None and self.font:
            win_text = ""
            if self._winner == 1:
                win_text = "黑棋 (P1/Agent) 胜!"
            elif self._winner == -1:
                win_text = "红棋 (P2/Opponent) 胜!"
            elif self._winner == 0:
                win_text = "平局!"
            text_surface = self.font.render(win_text, True, (0, 128, 0))
            text_rect = text_surface.get_rect(center=(self.screen_size // 2, 15))
            canvas.blit(text_surface, text_rect)
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """
        关闭环境。
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.font.quit()
            pygame.quit()
            self.window = None
