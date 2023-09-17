import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQNAgent:
    def __init__(self, model, state_dim, action_dim, learning_rate=1e-5, gamma=0.99, epsilon_decay=0.98):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=100)  # Replay buffer

    def select_action(self, state):
#         if np.random.rand() < self.epsilon:
        if False:
            p, r = random.randrange(self.action_dim[0]), random.randrange(self.action_dim[1])
#         q_values = self.model(torch.tensor(state[0], dtype=torch.double), torch.tensor(state[1], dtype=torch.double).view(1, 1))
#         print(random.randrange(self.action_dim[0]), random.randrange(self.action_dim[1]))
        p, r = 0, 0
        return p, r#torch.argmax(q_values[0]).item(), torch.argmax(q_values[1]).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        losses = []
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                r = self.model(
                        b=torch.tensor(next_state[0], dtype=torch.double),
                        n=torch.tensor(next_state[1], dtype=torch.double).view(-1,1))
                target[0] = reward[0] + self.gamma * torch.max(r[0])
                target[1] = reward[1] + self.gamma * torch.max(r[1])
            r = self.model(
                    b=torch.tensor(state[0], dtype=torch.double),
                    n=torch.tensor(state[1], dtype=torch.double).view(-1,1))
            target_q_values = (r[0].clone(), r[1].clone())
            target_q_values[0][action[0]] = target[0]
            target_q_values[1][action[1]] = target[1]

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(r[0], target_q_values[0]) + nn.MSELoss()(r[1], target_q_values[1])
            loss = loss * 1e3

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return np.mean(losses)