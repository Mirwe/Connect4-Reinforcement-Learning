import random

import torch

from DQN import DQN
from GPTopponent import GPTopponent
from MiniMaxOpponent import MinimaxOpponent

COLS=7

dqn = DQN().to("cpu")
dqn.load_state_dict(torch.load("runs/model1_02-11-24_23-30.pt"))


def agent_move(board):
    """Agent selects the column and inserts a coin"""
    t = torch.Tensor(board)
    flatten_state = torch.flatten(t)
    col = int(dqn(flatten_state).argmax())
    return col


def gpt_move(board, player_id):
    """GPT opponent selects the column and inserts a coin"""
    gpt = GPTopponent(board.copy(), player_id)
    col = gpt.ai_move()
    return col


def random_move():
    """Random opponent selects a random column and inserts a coin"""
    col = random.randint(0, COLS - 1)
    return col


def minimax_move(game, player_id):
    """Minimax opponent selects the move"""
    opponent = MinimaxOpponent(game, minimax_player_id=player_id)
    col = opponent.choose_action()
    return col
