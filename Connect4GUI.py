import random
import torch
import Opponents
import tkinter as tk
from tkinter import messagebox
from DQN import DQN
from Forza4 import Forza4
from GPTopponent import GPTopponent
from MiniMaxOpponent import MinimaxOpponent

# Constants
ROWS = 6
COLS = 7
CIRCLE_SIZE = 80  # Diameter of each circle
PLAYER_COLORS = ["#FF3333", "#FFD700"]  # Colors for Player 1 (Red) and Player 2 (Yellow)
BACKGROUND_COLOR = "#1E90FF"  # Blue background for Connect4 board

# Load DQN Agent if used
# Configure players as needed (e.g., "GPT", "Random", "Minimax", "Agent", "Human")

# Agent trained as player 1
players = ["Agent", "Human"]

# Initialize the game
game = Forza4()


class Connect4GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect4 Game with AI")
        self.root.configure(bg="white")

        self.current_player = random.randint(1, 2)  # Random start
        self.game = game
        self.board = [[None for _ in range(COLS)] for _ in range(ROWS)]

        # Create canvas for the board
        self.canvas = tk.Canvas(root, width=COLS * CIRCLE_SIZE, height=ROWS * CIRCLE_SIZE, bg=BACKGROUND_COLOR)
        self.canvas.grid(row=1, column=0, columnspan=COLS)

        # Draw the empty board
        for row in range(ROWS):
            for col in range(COLS):
                x1 = col * CIRCLE_SIZE
                y1 = row * CIRCLE_SIZE
                x2 = x1 + CIRCLE_SIZE
                y2 = y1 + CIRCLE_SIZE
                self.canvas.create_oval(x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill="white", tags=f"cell_{row}_{col}")

        # Status label
        self.status_label = tk.Label(root, text="Player 1's Turn", font=("Arial", 14), fg="black", bg="white")
        self.status_label.grid(row=2, column=0, columnspan=COLS, pady=10)

        # Column buttons for human player
        button_frame = tk.Frame(root, bg="white")
        button_frame.grid(row=0, column=0, columnspan=COLS, pady=5)
        self.buttons = []
        for col in range(COLS):
            button = tk.Button(button_frame, text="↓", font=("Arial", 20), width=4,
                               command=lambda c=col: self.play_turn(c))
            button.grid(row=0, column=col, padx=2, pady=2)
            self.buttons.append(button)

        # Start game loop for non-human player moves
        self.update_turn()

    def play_turn(self, col):
        """Handles the Human player's turn"""
        if players[self.current_player - 1] == "Human":
            self.insert_coin(self.current_player, col)

    def insert_coin(self, player, col):
        """Inserts a coin for the player in the specified column and updates the GUI"""
        row = next((r for r in range(ROWS - 1, -1, -1) if self.game.board[r][col] == 0), None)
        if row is not None:
            self.game.insert_coin(player, col)
            cell_id = f"cell_{row}_{col}"
            self.canvas.itemconfig(cell_id, fill=PLAYER_COLORS[player - 1])

            if self.game.check_victory(player):
                self.status_label.config(text=f"Player {player} Wins!")
                messagebox.showinfo("Game Over", f"Player {players[player - 1]} wins!")

            elif self.game.check_tie():
                self.status_label.config(text="It's a tie!")
                messagebox.showinfo("Game Over", "It's a tie!")

            else:
                self.switch_player()
        else:
            messagebox.showwarning("Column Full", "This column is full! Choose another.")

    def update_turn(self):
        """Handles the turn update loop for non-human players"""
        if players[self.current_player - 1] == "Human":
            # Enable buttons for human interaction
            for button in self.buttons:
                button.config(state=tk.NORMAL)
        else:
            # Disable buttons during AI's turn
            for button in self.buttons:
                button.config(state=tk.DISABLED)

            if players[self.current_player - 1] == "Agent":
                self.agent_move()
            elif players[self.current_player - 1] == "GPT":
                self.gpt_move()
            elif players[self.current_player - 1] == "Random":
                self.random_move()
            elif players[self.current_player - 1] == "Minimax":
                self.minimax_move()

        # Wait a moment and call itself again to keep updating AI turns
        self.root.after(500, self.update_turn)

    def agent_move(self):
        """Agent selects the column and inserts a coin"""
        col = Opponents.agent_move(self.game.board)
        self.insert_coin(self.current_player, col)

    def gpt_move(self):
        """GPT opponent selects the column and inserts a coin"""
        col = Opponents.gpt_move(self.game.board, self.current_player)
        self.insert_coin(self.current_player, col)

    def random_move(self):
        """Random opponent selects a random column and inserts a coin"""
        col = Opponents.random_move()
        self.insert_coin(self.current_player, col)

    def minimax_move(self):
        """Random opponent selects a random column and inserts a coin"""
        col = Opponents.minimax_move(self.game, self.current_player)
        self.insert_coin(self.current_player, col)

    def switch_player(self):
        """Switches to the other player"""
        self.current_player = 1 if self.current_player == 2 else 2
        self.status_label.config(
            text=f"{players[self.current_player - 1]} Turn ({'Red' if self.current_player == 1 else 'Yellow'})")


# Main GUI loop
root = tk.Tk()
gui = Connect4GUI(root)
root.mainloop()
