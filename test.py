import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from tetris import Tetris
from tetrismaster import TetrisMaster


if __name__ == '__main__':
	w, h = 10, 20
	epoch = 1000

	game = Tetris(w, h)
	model = TetrisMaster(bw=w, bh=h).double()

	cp = torch.load('checkpoints/test_model')
	model.load_state_dict(cp['model_state_dict'])

	for _ in range(100):
		# x = np.concatenate([game.board.reshape(-1), [game.next]])
		x = torch.tensor(game.board, dtype=torch.double).unsqueeze(0)
		y = model(x, torch.tensor(game.next, dtype=torch.double).view(1, 1))#[0]
		game.clear_board()
		game.set_random_state()
		with torch.no_grad():
			col = y[:w].argmax()
			rot = y[w:].argmax()
			res = game.turn(col, rot)
		game.print()
		print(f'{res=}')
		time.sleep(2)

