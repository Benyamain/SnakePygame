import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# To save our model
import os

class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Input, output
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Apply the activation function on the linear layer
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x
    
    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:

    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        # Optimization step
        self.optimizer = optim.Adam(model.parameters(), learning_rate = self.learning_rate)
        # Criterion: To track the loss (a loss function)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype = torch.float)

        # One dimension such as (1, x)
        if len(state.shape) == 1:
            # Appends one dimension in the beginning
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            # A tuple with one value
            game_over = (game_over, )
