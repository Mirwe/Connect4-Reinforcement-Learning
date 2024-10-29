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
        for i in reversed(range(self.num_rows)):
            if self.board[i][col] == 0:
                self.board[i][col] = player
                return True

        return False

    def check_win(self):

        for row in self.board:
            for i in range(0, self.num_cols):
                if i+4 > self.num_cols:
                    break

                if row[i] != 0 and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    print("YOU WIN")
                    return True
