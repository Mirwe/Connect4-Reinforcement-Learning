import random
import torch
import Opponents



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

        # self.dqn_baseline = DQN_OLD(42, 7).to("cpu")
        # self.dqn_baseline.load_state_dict(torch.load("runs\\baseline.pt"))

    def __str__(self):
        o = ""
        for row in self.board:
            o += str(row) + "\n"

        return o

    def col_is_valid(self, col):
        return self.board[0][col] == 0

    def insert_coin(self, player, col, return_row=False):

        if not self.col_is_valid(col):
            if return_row:
                return -1
            return False

        if col >= self.num_cols:
            if return_row:
                return -1
            return False

        for i in reversed(range(self.num_rows)):
            if self.board[i][col] == 0:
                self.board[i][col] = player
                if return_row:
                    return i
                return True

        if return_row:
            return -1
        return False

    def remove_coin(self, row, col):

        if not self.col_is_valid(col):
            return False

        if col >= self.num_cols:
            return False

        self.board[row][col] = 0
        return True

    def check_victory(self, player, number_of_coins=4):

        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if self.board[r][c] == player:
                    # Controllo verso destra
                    if c + 3 < self.num_cols and all(self.board[r][c + i] == player for i in range(number_of_coins)):
                        return True
                    # Controllo verso il basso
                    if r + 3 < self.num_rows and all(self.board[r + i][c] == player for i in range(number_of_coins)):
                        return True
                    # Controllo diagonale verso destra in basso
                    if r + 3 < self.num_rows and c + 3 < self.num_cols \
                            and all(self.board[r + i][c + i] == player for i in range(number_of_coins)):
                        return True
                    # Controllo diagonale verso sinistra in basso
                    if r + 3 < self.num_rows and c - 3 >= 0 \
                            and all(self.board[r + i][c - i] == player for i in range(number_of_coins)):
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
        if col + 3 < self.num_cols and all(self.board[row][col + i + 1] == 2 for i in range(3)):
            return True

        if col - 3 >= 0 and all(self.board[row][col - (j + 1)] == 2 for j in range(3)):
            return True

        # Controllo verso il basso
        if row + 3 < self.num_rows and all(self.board[row + i + 1][col] == 2 for i in range(3)):
            return True

        # Controllo diagonale verso destra in basso
        if row + 3 < self.num_rows and col + 3 < self.num_cols \
                and all(self.board[row + i + 1][col + i + 1] == 2 for i in range(3)):
            return True

        # Controllo diagonale verso sinistra in basso
        if row + 3 < self.num_rows and col - 3 >= 0 \
                and all(self.board[row + i + 1][col - i - 1] == 2 for i in range(3)):
            return True

        return False

    def check_consecutive_coins(self, col, player=2, number_of_coins=4):
        # 1 Agent
        # 2 Opponent

        row = 0
        for i, a in enumerate(self.board):
            if a[col] == 1:
                row = i
                break

        # Check horizontally for blocking opponent's win
        count_left = 0
        for i in range(1, number_of_coins):
            if col - i < 0 or self.board[row][col - i] != player:
                break
            count_left += 1

        count_right = 0
        for i in range(1, number_of_coins):
            if col + i >= self.num_cols or self.board[row][col + i] != player:
                break
            count_right += 1

        if count_left + count_right >= number_of_coins - 1:
            return True

        # Check vertically
        count_down = sum(
            1 for i in range(1, number_of_coins) if row + i < self.num_rows and self.board[row + i][col] == player)
        if count_down >= number_of_coins - 1:
            return True

        # Check diagonal verso giu e destra
        count_diag_down = 0
        for i in range(1, number_of_coins - 1):
            if col + i >= self.num_cols or row + i >= self.num_rows or self.board[row + i][col + i] != player:
                break
            count_diag_down += 1

        count_diag_up = 0
        for i in range(1, number_of_coins):
            if col - i < 0 or row - i < 0 or self.board[row - i][col - i] != player:
                break
            count_diag_up += 1

        if count_diag_down + count_diag_up >= number_of_coins - 1:
            return True

        # check diagonally verso giu sinistra
        count_diag1_down = 0
        for i in range(1, number_of_coins):
            if col - i < 0 or row + i >= self.num_rows or self.board[row + i][col - i] != player:
                break
            count_diag1_down += 1

        count_diag1_up = 0
        for i in range(1, number_of_coins):
            if col + i >= self.num_cols or row - i < 0 or self.board[row - i][col + i] != player:
                break
            count_diag1_up += 1

        if count_diag1_down + count_diag1_up >= number_of_coins - 1:
            return True

        return False

    # Return rewards after the move
    # Board, New State, Reward, terminate, playerWin
    def make_move(self, player, col, opponent="GPT"):

        correctly_inserted = self.insert_coin(player, col)

        new_state = torch.flatten(torch.Tensor(self.board))

        reward = -0.05
        # reward = 0

        # piccolo reward se inizia al centro
        if 2 <= col < 5 and all(self.board[i][col] == 0 for i in range(5)):
            reward += 0.1

        if not correctly_inserted:
            return self.board, new_state, -0.5, True, None

        if self.check_victory(player):
            return self.board, new_state, 1, True, player

        if self.check_tie():
            return self.board, new_state, 0.5, True, None

        # Reward Agent if it has blocked the opponent's win
        if self.check_consecutive_coins(col):  # if self.agent_move_blocks_win(col):
            reward += 0.25

        # Reward Agent if it has put 3 coins near
        if self.check_consecutive_coins(col, player=1, number_of_coins=3):  # if self.agent_move_blocks_win(col):
            reward += 0.15

        # Trained againt GPT opponent
        if opponent == "GPT":
            col = Opponents.gpt_move(self.board.copy(), 2)

        elif opponent == "Random":
            col = Opponents.random_move()

        elif opponent == "Human":
            print(self)
            col = int(input("Enter column [0-6]: "))

        elif opponent == "Agent":
            self.switch_coins()
            Opponents.agent_move(self.board)
            self.switch_coins()

        elif opponent == "Minimax":
            col = Opponents.minimax_move(self, 2)

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
