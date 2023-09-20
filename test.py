import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from tetris import Tetris
from tetris_master import TetrisMaster2 as TetrisMaster


if __name__ == '__main__':
    import os
    w, h = 10, 20
    epoch = 1000

    game = Tetris(w, h)
    model = TetrisMaster().double()

    cp = torch.load('checkpoints/test_model')
    model.load_state_dict(cp['model_state_dict'])

    while 1:
        # game.reset()
        os.system('clear')
        x = torch.tensor(game.board, dtype=torch.double)
        y = model(x, torch.tensor(game.next, dtype=torch.double).view(1, 1))#[0]
        with torch.no_grad():
            print(f'probs {y.detach().numpy()}')
            a = y.argmax()
            res = game.step(a)
        # game.print()
        print(game)
        time.sleep(0.1)

