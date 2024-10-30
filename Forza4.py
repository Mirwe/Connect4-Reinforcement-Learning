import numpy as np


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

    def __str__(self):
        o = ""
        for row in self.board:
            o += str(row) + "\n"

        return o

    def insert_coin(self, player, col):
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
