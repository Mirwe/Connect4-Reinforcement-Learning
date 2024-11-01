import datetime
import itertools
import math
import os
import random

import numpy as np
import torch.optim as optim
import torch
import matplotlib
import matplotlib.pyplot as plt
from torch import nn
from datetime import datetime, timedelta

from experience_replay import ReplayMemory
from DQN import DQN
from Forza4 import Forza4
import yaml

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


class Agent:

    def __init__(self):
        with open('hyperparameters.yml') as file:
            hyperparameters = yaml.safe_load(file)

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.num_episodes = hyperparameters['num_episodes']

        self.loss_fn = nn.MSELoss()  # Mean Square Error
        self.optimizer = None

        # Path to Run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'log1.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'model1.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'graph1.png')

        self.LOG_FILE2 = os.path.join(RUNS_DIR, f'log_model2.log')
        self.MODEL_FILE2 = os.path.join(RUNS_DIR, f'model_model2.pt')
        self.GRAPH_FILE2 = os.path.join(RUNS_DIR, f'graph_model2.png')

    def train(self):
        num_states = 42
        num_action = 7

        start_time = datetime.now()
        last_graph_update_time = start_time

        rewards_per_episode_player1 = []
        # rewards_per_episode_player2 = []

        policy_dqn = DQN().to(device)
        # Load learned policy
        policy_dqn.load_state_dict(torch.load("runs/model1_to_beat.pt"))

        memory_1 = ReplayMemory(10000)
        # memory_2 = ReplayMemory(10000)
        epsilon_1 = self.epsilon_init
        # epsilon_2 = self.epsilon_init
        target_dqn = DQN().to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # target2_dqn = DQN(num_states, num_action).to(device)
        # target2_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        epsilon_1_history = []
        step_count = 0
        best_reward_1 = -999999999

        agent_wins = 0
        opponent_wins = 0

        for episode in range(0, self.num_episodes):
            game = Forza4()
            state_1 = torch.flatten(torch.Tensor(game.board))
            terminated = False

            episode_reward_player_1 = 0.0

            while not terminated:

                if random.random() < epsilon_1:
                    action_player1 = random.randint(0, 6)
                else:
                    action_player1 = int(policy_dqn(state_1).argmax())

                new_board, new_state, reward1, terminated, player_win = game.make_move(1, action_player1, "Agent")

                if player_win == 1:
                    agent_wins += 1
                elif player_win == 2:
                    opponent_wins += 1

                episode_reward_player_1 += reward1
                game.board = new_board

                action_player1 = torch.tensor(action_player1, dtype=torch.int64, device=device)
                reward1 = torch.tensor(reward1, dtype=torch.float, device=device)
                memory_1.append((state_1, action_player1, new_state, reward1, terminated))

                if terminated:
                    break

                state_1 = new_state

            rewards_per_episode_player1.append(reward1)

            if episode_reward_player_1 > best_reward_1:
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward 1{episode_reward_player_1:0.1f}) at episode {episode}, saving model..."
                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward_1 = episode_reward_player_1

            if episode % 500 == 0 and episode != 0:
                print(f"Episode {episode}, Total Reward: {episode_reward_player_1}, Epsilon: {epsilon_1}, "
                      f"Agent Win Rate {agent_wins / (agent_wins + opponent_wins)}")
                torch.save(policy_dqn.state_dict(), "model1_checkpoint.pt")

                agent_wins = 0
                opponent_wins = 0

            # Update graph every x seconds
            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=10):
                try:
                    self.save_graph(rewards_per_episode_player1, epsilon_1_history)
                except:
                    print("fail")
                last_graph_update_time = current_time

            # if enough experience
            if len(memory_1) > self.mini_batch_size:
                mini_batch_1 = memory_1.sample(self.mini_batch_size)

                self.optimize(mini_batch_1, policy_dqn, target_dqn)

                epsilon_1 = max(epsilon_1 * self.epsilon_decay, self.epsilon_min)

                # epsilon decay based on performance (increase exploration when losing, decrease when winning)
                # if reward1 >= -0.5:
                #     epsilon_1 = max(epsilon_1 * self.epsilon_decay, self.epsilon_min)
                # else:
                #     epsilon_1 = min(epsilon_1 * 1.001, 1)

                # Decay epsilon

                epsilon_1_history.append(epsilon_1)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    # target2_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # for state, action, new_state, reward, terminated in mini_batch:
        #     if terminated:
        #         target = reward
        #
        #     else:
        #         with torch.no_grad():
        #             target_q = reward + self.discount_factor_g * target_dqn(new_state).max()

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # if self.enable_double_dqn:
            best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

            target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                       target_dqn(new_states).gather(dim=1,
                                                     index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            # else:
            # Calculate target Q values (expected returns)
            # target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update network parameters i.e. weights and biases

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


if __name__ == '__main__':
    a = Agent()
    a.train()
