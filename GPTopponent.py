import random


class GPTopponent:

    def __init__(self, board, gpt_player_number):
        self.board = board.copy()
        self.num_rows = 6
        self.num_cols = 7
        self.gpt_player_number = gpt_player_number
        self.opponenent_player_number = 2 if gpt_player_number == 1 else 1

    def make_move(self, col):
        # Check for valid move
        if col < 0 or col >= self.num_cols or self.board[0][col] != 0:
            return -1

        # Place the coin in the first empty row from the bottom
        for row in range(self.num_rows - 1, -1, -1):
            if self.board[row][col] == 0:
                return col

        return -1

    def check_win(self, player):
        # Check all directions for a win (horizontal, vertical, diagonal)
        # Horizontal
        for row in range(self.num_rows):
            for col in range(self.num_cols - 3):
                if all(self.board[row][col + i] == player for i in range(4)):
                    return True
        # Vertical
        for col in range(self.num_cols):
            for row in range(self.num_rows - 3):
                if all(self.board[row + i][col] == player for i in range(4)):
                    return True
        # Diagonal (/)
        for row in range(3, self.num_rows):
            for col in range(self.num_cols - 3):
                if all(self.board[row - i][col + i] == player for i in range(4)):
                    return True
        # Diagonal (\)
        for row in range(self.num_rows - 3):
            for col in range(self.num_cols - 3):
                if all(self.board[row + i][col + i] == player for i in range(4)):
                    return True
        return False

    def get_valid_moves(self):
        # Return a list of columns that are valid moves
        return [col for col in range(self.num_cols) if self.board[0][col] == 0]

    def undo_move(self, col):
        # Remove the top coin from the column
        for row in range(self.num_rows):
            if self.board[row][col] != 0:
                self.board[row][col] = 0
                return

    def ai_move(self):
        # Check if AI can win with any move
        for move in self.get_valid_moves():
            if self.simulate_move(move, self.gpt_player_number):  # Check AI's winning move
                return move

        # Check if player 1 can win with any move, so we can block it
        for move in self.get_valid_moves():
            if self.simulate_move(move, self.opponenent_player_number):  # Check Player 1's winning move
                return move

        # Otherwise, pick a random valid move
        return random.choice(self.get_valid_moves())

    def simulate_move(self, col, player):
        # Simulate placing a piece in the column and check if it leads to a win
        row = self.get_next_open_row(col)
        if row is not None:
            self.board[row][col] = player
            win = self.check_win(player)
            self.board[row][col] = 0  # Undo the simulated move
            return win
        return False

    def get_next_open_row(self, col):
        # Find the next available row in a column (topmost empty slot)
        for row in range(self.num_rows - 1, -1, -1):
            if self.board[row][col] == 0:
                return row
        return None
