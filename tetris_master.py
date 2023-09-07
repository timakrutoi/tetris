import numpy as numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class TetrisMaster(nn.Module):
	def __init__(self, n_layers=2, bw=10, bh=20, c_hidden=32):
		super(TetrisMaster, self).__init__()
		self.eps = 0.9
		self.bw = bw
		self.bh = bh
		kernel = (2, 2)

		self.start = nn.Linear(1, c_hidden)
		self.np = nn.Linear(c_hidden, c_hidden)
		self.pipeline = nn.Sequential()

		self.pipeline.append(nn.Conv2d(1, c_hidden, kernel, stride=1, padding=1))

		for i in range(n_layers):
			self.pipeline.append(nn.Conv2d(c_hidden, c_hidden, kernel, stride=1, padding=1))

		self.pool = nn.MaxPool2d(2)
		self.act = nn.ReLU()
		# 6 coz its quantity of tetriminos
		self.end1 = nn.Linear(96*2, bw)
		self.end2 = nn.Linear(96*2, 4)
		# self.pipeline.append(nn.ReLU())
		# self.pipeline.append(nn.Softmax(dim=0))

	def forward(self, b, n):
		x = b
		n = self.start(n)
		for m in self.pipeline:
			x1 = self.pool(m(x))
			# x1 = m(x)
			x2 = self.np(n)[0]
			x2 = x2.view(-1, 1, 1).expand(*x1.shape)
			# print(f'{x1.shape=}, {x2.shape=}')
			x = self.act(x1 + x2)

		x1, x2 = self.end1(x.view(-1)), self.end2(x.view(-1))
		# print(f'{x1.shape=} {x2.shape=}')
		x = torch.cat((x1, x2), dim=0)
		return x

	def select_action(self):
		y = self.forward(x)
		col = y[:w]
		rot = y[w:]

		if random.random() > self.eps:
			col = random.randint(0, self.bw)
			rot = random.randint(0, 4)
			return torch.cat(F.one_hot(col, self.bw), F.one_hot(rot, 4))
		
		return torch.cat(F.one_hot(col, self.bw), F.one_hot(rot, 4))


if __name__ == '__main__':
	pass


