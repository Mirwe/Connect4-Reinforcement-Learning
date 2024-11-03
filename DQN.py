"""Deep Q Learning"""
import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q Network (DQN) model for playing the game.

    The model consists of convolutional layers (Conv2d) and fully connected layers (Linear)
    to process the input state and output Q-values for each action.

    Attributes:
    - conv1: First convolutional layer with 64 output channels.
    - conv2: Second convolutional layer with 128 output channels.
    - conv3: Third convolutional layer with 256 output channels.
    - fc1: First fully connected layer with 256 neurons.
    - fc2: Second fully connected layer with 128 neurons.
    - fc3: Third fully connected layer with 128 neurons.
    - fc4: Fourth fully connected layer with 7 neurons (one for each column).

    Methods:
    - forward(x): Processes the input state through the convolutional and fully connected layers
                  to produce Q-values for each action.
    """

    def __init__(self):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)  # Keep the same size
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Keep the same size
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Keep the same size

        # Fully connected layers after flattening
        self.fc1 = nn.Linear(256 * 6 * 7, 256)  # Adjust the input size based on output from conv layers
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 7)  # 7 actions, one for each column

    def forward(self, x):
        x = x.view(-1, 1, 6, 7)  # Reshape input to match Conv2D expectations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten before fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)  # Q-values for each column


if __name__ == '__main__':
    state_dim = 42
    action_dim = 7
    net = DQN(state_dim, action_dim)
    initial_state = []

    for i in range(0, 6):
        initial_state.append([0] * 7)
    t = torch.Tensor(initial_state)
    flatten_state = torch.flatten(t)
    output = int(net(flatten_state).argmax())
    print(output)
