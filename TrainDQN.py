import math
import random

import numpy as np
import torch.optim as optim
import torch
import DQN

# https://github.com/neoyung/connect-4/blob/master/.ipynb_checkpoints/connect_X-checkpoint.ipynb

# epilson decay graph
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

n_actions = 7

policy_net = DQNAgent.DQN(n_actions).to(device)
# target_net will be updated every n episodes to tell policy_net a better estimate of how far off from convergence
target_net = DQNAgent.DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
# set target_net in testing mode
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())


def select_action(state, available_actions, steps_done=None, training=True):
    # batch and color channel
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(dim=0).unsqueeze(dim=0)
    epsilon = random.random()
    if training:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    else:
        eps_threshold = 0

    # follow epsilon-greedy policy
    if epsilon > eps_threshold:
        with torch.no_grad():
            # action recommendations from policy net
            r_actions = policy_net(state)[0, :]
            state_action_values = [r_actions[action] for action in available_actions]
            argmax_action = np.argmax(state_action_values)
            greedy_action = available_actions[argmax_action]
            return greedy_action
    else:
        return random.choice(available_actions)


