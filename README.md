# Connect 4 Reinforcement Learning Agent
Repository for learning purposes.

This repository contains an agent trained to play the game Connect 4 using the Deep Q-Learning (DQN) algorithm. 
The agent uses a neural network to approximate the Q-values for each state-action pair, and it learns by interacting with the game environment and updating its policy based on the experiences gained.

## Files Included

- `Connect4GUI.py`: The game environment for Connect4, which provides a simple graphical user interface for visualizing the game board and player actions. Here it is possible to play against different AI
  - GPT: A simple AI created by asking to GPT
  - Agent: The DQN Agent trained with `Agent.py`
  - Random
  - Minimax algorithm
- `Agent.py`: The main script that trains the agent using the DQN algorithm. It can be trained with all the the above AIs
- `DQN.py`: The implementation of the deep neural network used as the policy and target models.
- `experience_replay.py`: The implementation of the experience replay memory used to store and sample experiences.
- `Forza4.py`: The game environment for Connect 4, which provides the necessary functions to interact with the game.
- `hyperparameters.yml`: A configuration file containing the hyperparameters used for training the agent.
- `runs/`: A directory where the agent's training logs, model checkpoints, and graphs are saved.

