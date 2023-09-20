import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQNAgent:
    def __init__(self, model, state_dim, action_dim, learning_rate=5e-3, gamma=0.99, epsilon_decay=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=2000)  # Replay buffer

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        q_values = self.model(
                torch.tensor(state[0], dtype=torch.double),
                torch.tensor(state[1], dtype=torch.double).view(-1, 1))
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = target + self.gamma * torch.max(self.model(
                    torch.tensor(state[0], dtype=torch.double),
                    torch.tensor(state[1], dtype=torch.double).view(-1, 1)))

            target_q_values = self.model(
                torch.tensor(state[0], dtype=torch.double),
                torch.tensor(state[1], dtype=torch.double).view(-1, 1)).clone()

            target_q_values[action] = target 
            loss = nn.MSELoss()(self.model(
                torch.tensor(state[0], dtype=torch.double),
                torch.tensor(state[1], dtype=torch.double).view(-1, 1)
            ), target_q_values) * 1e2


            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

        return loss
