import random


class MinimaxOpponent:
    def __init__(self, game, minimax_player_id=2, max_depth=3):
        self.game = game
        self.board = [row[:] for row in game.board]  # Copy initial board state
        self.minimax_player_id = minimax_player_id  # AI player ID
        self.opponent_id = 1 if minimax_player_id == 2 else 2
        self.max_depth = max_depth

    def minimax(self, depth, maximizing_player):
        """
        Perform the minimax algorithm to evaluate the best possible move.

        This function recursively explores the game tree to a specified depth,
        evaluating the board state from the perspective of the AI player and
        the opponent. It aims to maximize the AI's score while minimizing the
        opponent's score.

        Parameters:
        depth (int): The current depth in the game tree. The algorithm will
                     stop exploring further when depth reaches 0.
        maximizing_player (bool): A boolean indicating whether the current
                                  turn is for the maximizing player (AI) or
                                  the minimizing player (opponent).

        Returns:
        int: The evaluation score of the board state. A positive score
             indicates a favorable state for the AI, a negative score
             indicates a favorable state for the opponent, and zero
             indicates a neutral or draw state.
        """
        # Check terminal states for wins, losses, or draw
        if self.game.check_victory(self.minimax_player_id):  # AI's win
            return 1
        elif self.game.check_victory(self.opponent_id):  # Opponent's win
            return -1
        elif self.game.check_tie() or depth == 0:  # Draw or depth limit reached
            return 0

        valid_moves = [col for col in range(self.game.num_cols) if self.game.col_is_valid(col)]

        if maximizing_player:  # AI's turn
            max_eval = float('-inf')
            for col in valid_moves:
                row = self.game.insert_coin(self.minimax_player_id, col, return_row=True)
                if row != -1:  # Ensure a valid move
                    eval = self.minimax(depth - 1, False)  # Switch to minimizing player
                    self.game.remove_coin(row, col)  # Undo move
                    max_eval = max(max_eval, eval)
            return max_eval
        else:  # Opponent's turn
            min_eval = float('inf')
            for col in valid_moves:
                row = self.game.insert_coin(self.opponent_id, col, return_row=True)
                if row != -1:  # Ensure a valid move
                    eval = self.minimax(depth - 1, True)  # Switch to maximizing player
                    self.game.remove_coin(row, col)  # Undo move
                    min_eval = min(min_eval, eval)
            return min_eval

    def choose_action(self):
        best_score = float('-inf')
        best_move = random.choice([col for col in range(self.game.num_cols) if self.game.col_is_valid(col)])

        # Non iniziare sempre dalla colonna 0
        cols = [i for i in range(self.game.num_cols)]
        random.shuffle(cols)

        for col in cols:
            if self.game.col_is_valid(col):  # Check that the column is valid
                row = self.game.insert_coin(self.minimax_player_id, col, return_row=True)
                if row != -1:
                    score = self.minimax(self.max_depth - 1, False)
                    self.game.remove_coin(row, col)  # Undo the move
                    if score > best_score:
                        best_score = score
                        best_move = col

        return best_move
