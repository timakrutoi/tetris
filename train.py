#!/usr/bin/python3
#encode=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from tetris import Tetris
from tetris_master import TetrisMaster2
from Agent import DQNAgent


if __name__ == '__main__':
    w, h = 10, 20
    game_len = 50
    epoch = 10000
    betta = 0.3
    gamma = 0.89
    rf = 4
    lr = 1e-5

    env = Tetris(w, h)

    # Initialize the agent
    state_dim = (env.board.shape, 1)
    action_dim = 4
    batch_size = 32

    losses = []

    model = TetrisMaster2().double()

    agent = DQNAgent(model, state_dim, action_dim)

    for e in range(100):
        # Training the DQN agent
        episodes = 200
        torch.autograd.set_detect_anomaly(True)
        it = tqdm(range(episodes))
        for episode in it:
            with torch.no_grad():
                state = env.reset()
                total_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    # print(state[0].shape, next_state[0].shape)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward = total_reward + reward
            loss = agent.replay(batch_size)
            it.set_postfix(str=f"e: {episode + 1}, loss {loss.item():.1e}, rwd: {round(total_reward, rf)}, eps {round(agent.epsilon, rf)}")
        it.close()

        # Evaluate the trained agent
        test_episodes = 10
        it = tqdm(range(test_episodes))
        for _ in it:
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                state = next_state
                total_reward += reward
            it.set_postfix(str=f"Test Episode, loss {round(loss.item(), rf)}, Total Reward: {round(total_reward, rf)}")

        print('saving')
        try:
            torch.save({
                'model_state_dict': agent.model.state_dict(),
            }, 'checkpoints/test_model')
        except KeyboardInterrupt:
            torch.save({
                'model_state_dict': agent.model.state_dict(),
            }, 'checkpoints/test_model')

        it.close()
