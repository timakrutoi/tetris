import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import copy
from collections import deque


class DQNAgent:
    def __init__(self, model, state_dim, action_dim, learning_rate, betta, gamma, epsilon, epsilon_min, epsilon_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.betta = betta
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.online_net = model
        self.target_net = copy.deepcopy(self.online_net)
        self.update_rat = 5000
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=2000)  # Replay buffer

    def select_action(self, state, test=False):
        if not test and np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
            # return random.randrange(2)
        q_values = self.online_net(
                torch.tensor(state[0], dtype=torch.double),
                torch.tensor(state[1], dtype=torch.double).view(-1, 1))
        return torch.argmax(q_values).item()
        # return torch.multinomial(q_values, 1, replacement=True)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            raise Exception(f'Memory size is less than batch size ({len(self.memory)} < {batch_size})')
        minibatch = random.sample(self.memory, batch_size)
        target_q_values = None
        ret_loss = []
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = target + self.gamma * torch.max(self.online_net(
                    torch.tensor(next_state[0], dtype=torch.double),
                    torch.tensor(next_state[1], dtype=torch.double).view(-1, 1)))

            target_q_values = self.target_net(
                torch.tensor(state[0], dtype=torch.double),
                torch.tensor(state[1], dtype=torch.double).view(-1, 1)).clone()
            # print(target_q_values)

            target_q_values[action] = target 
            ys = self.online_net(
                torch.tensor(state[0], dtype=torch.double),
                torch.tensor(state[1], dtype=torch.double).view(-1, 1))
            
            loss = nn.MSELoss()(ys, target_q_values)
            entropy = self.betta * ((ys + 1e-9).log() * ys).sum() / 10
            # print(loss, entropy)
            loss = loss - entropy
            loss = loss * 100
            ret_loss.append(loss.item())
            # print(ys)
            # print(target_q_values)
            # print('-'*30)

            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
            if i % self.update_rat:
                self.target_net = copy.deepcopy(self.online_net)

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

        return np.mean(ret_loss)
