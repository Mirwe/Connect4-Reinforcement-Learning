import random

import torch
from DQN import DQN
from Forza4 import Forza4

import os

from GPTopponent import GPTopponent

import tkinter as tk

# Constants
ROWS = 6
COLS = 7
CIRCLE_SIZE = 80  # Diameter of each circle
PLAYER_COLORS = ["#FF3333", "#FFD700"]  # Colors for Player 1 (Red) and Player 2 (Yellow)
BACKGROUND_COLOR = "#1E90FF"  # Blue background for Connect4 board


# Agent trainato come 1

players = ["Agent", "Human"]

if "Agent" in players:
    dqn = DQN().to("cpu")
    dqn.load_state_dict(torch.load("runs\\model1.pt"))


game = Forza4()
print(game)

player = random.randint(1, 2)

while True:

    print("Player " + str(player))

    col = -1

    if players[player-1] == "Human":
        col = input("Enter column [0-6]: ")

    if players[player-1] == "Random":
        col = random.randint(0, 6)

    if players[player-1] == "GPT":
        gpt = GPTopponent(game.board.copy(), player)
        col = gpt.ai_move()

    if players[player-1] == "Agent":
        t = torch.Tensor(game.board)
        flatten_state = torch.flatten(t)
        col = int(dqn(flatten_state).argmax())

    correctly_inserted = game.insert_coin(player=player, col=int(col))

    while not correctly_inserted:
        print("Not possible to insert coin there, retry ")
        col = input("Enter column [0-6]: ")
        correctly_inserted = game.insert_coin(player=player, col=int(col))

    win = game.check_victory(player)

    os.system('cls')
    print(game)

    if win:
        finished = True
        print("Player "+players[player-1]+" WIN")
        break

    if game.check_tie():
        finished = True
        print("IT'S TIE!")
        break

    if player == 2:
        player = 1
    else:
        player = 2


