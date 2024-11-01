import random

import numpy as np
import torch

from DQN import DQN, DQN_OLD
from GPTopponent import GPTopponent


class Forza4:

    def __init__(self):
        self.board = []
        self.num_rows = 6
        self.num_cols = 7
        self.players = [1, 2]

        # crea board
        # [[0, 0, 0, 0, 0 ,0 ,0]
        # ..
        # [0, 0, 0, 0, 0 ,0 ,0]]
        for i in range(0, self.num_rows):
            self.board.append([0] * self.num_cols)

        #self.dqn_baseline = DQN_OLD(42, 7).to("cpu")
        #self.dqn_baseline.load_state_dict(torch.load("runs\\baseline.pt"))

    def __str__(self):
        o = ""
        for row in self.board:
            o += str(row) + "\n"

        return o

    def col_is_valid(self, col):
        return self.board[0][col] == 0

    def insert_coin(self, player, col):

        if not self.col_is_valid(col):
            return False

        if col >= self.num_cols:
            return False

        for i in reversed(range(self.num_rows)):
            if self.board[i][col] == 0:
                self.board[i][col] = player
                return True

        return False

    def check_victory(self, player):

        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if self.board[r][c] == player:
                    # Controllo verso destra
                    if c + 3 < self.num_cols and all(self.board[r][c + i] == player for i in range(4)):
                        return True
                    # Controllo verso il basso
                    if r + 3 < self.num_rows and all(self.board[r + i][c] == player for i in range(4)):
                        return True
                    # Controllo diagonale verso destra in basso
                    if r + 3 < self.num_rows and c + 3 < self.num_cols \
                            and all(self.board[r + i][c + i] == player for i in range(4)):
                        return True
                    # Controllo diagonale verso sinistra in basso
                    if r + 3 < self.num_rows and c - 3 >= 0 \
                            and all(self.board[r + i][c - i] == player for i in range(4)):
                        return True
        return False

    # Return true if tie
    def check_tie(self):
        return 0 not in self.board[0]

    def swipe(self, fro, to):
        # i 2 devono diventare 1 e viceversa
        for i, r in enumerate(self.board):
            for j, c in enumerate(r):
                if c == fro:
                    r[j] = to

    def switch_coins(self):
        self.swipe(1, 3)
        self.swipe(2, 1)
        self.swipe(3, 2)

    def agent_move_blocks_win(self, col):
        # 1 Agent
        # 2 Opponent
        row = 0
        for i, a in enumerate(self.board):
            if a[col] == 1:
                row = i
                break

        # Controllo verso destra
        if col + 3 < self.num_cols and all(self.board[row][col + i+1] == 2 for i in range(3)):
            return True

        if col - 3 >= 0 and all(self.board[row][col - (j+1)] == 2 for j in range(3)):
            return True

        # Controllo verso il basso
        if row + 3 < self.num_rows and all(self.board[row + i+1][col] == 2 for i in range(3)):
            return True

        # Controllo diagonale verso destra in basso
        if row + 3 < self.num_rows and col + 3 < self.num_cols \
                and all(self.board[row + i+1][col + i+1] == 2 for i in range(3)):
            return True

        # Controllo diagonale verso sinistra in basso
        if row + 3 < self.num_rows and col - 3 >= 0 \
                and all(self.board[row + i+1][col - i-1] == 2 for i in range(3)):
            return True

        return False

    # Return rewards after the move
    # Board, New State, Reward, terminate, playerWin
    def make_move(self, player, col, opponent="GPT"):

        correctly_inserted = self.insert_coin(player, col)

        new_state = torch.flatten(torch.Tensor(self.board))

        reward = -0.05

        # piccolo reward se inizia al centro
        if 2 <= col < 5 and all(self.board[i][col] == 0 for i in range(5)):
            reward += 0.1

        if not correctly_inserted:
            return self.board, new_state, -0.2, True, None

        if self.check_victory(player):
            return self.board, new_state, 1, True, player

        if self.check_tie():
            return self.board, new_state, 0.5, True, None

        # Reward Agent if it has blocked the opponent's win
        if self.agent_move_blocks_win(col):
            reward += 0.35

        # Trained againt GPT opponent
        if opponent == "GPT":
            gpt = GPTopponent(self.board.copy(), 2)
            col = gpt.ai_move()

        if opponent == "Random":
            col = random.randint(0, 6)

        if opponent == "Human":
            print(self)
            col = int(input("Enter column [0-6]: "))

        if opponent == "Agent":
            self.switch_coins()
            t = torch.Tensor(self.board)
            flatten_state = torch.flatten(t)
            dqn = DQN().to("cpu")
            dqn.load_state_dict(torch.load("runs\\model1_to_beat.pt"), strict=False)
            col = int(dqn(flatten_state).argmax())
            self.switch_coins()

        correctly_inserted = self.insert_coin(2, col)
        while not correctly_inserted:
            col = random.randint(0, 6)
            correctly_inserted = self.insert_coin(2, col)

        new_state = torch.flatten(torch.Tensor(self.board))

        if self.check_victory(2):
            return self.board, new_state, -1, True, 2

        if self.check_tie():
            return self.board, new_state, 0.5, True, None

        return self.board, new_state, reward, False, None
