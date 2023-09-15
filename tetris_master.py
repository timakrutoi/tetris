import numpy as numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class GlobalPool2d(nn.Module):
    def __init__(self):
        super(GlobalPool2d, self).__init__()

    def forward(self, x):
        b, c, w, h = x.shape
        return nn.functional.adaptive_avg_pool2d(x, 1).reshape((b, c, 1, 1))


class TetrisMaster2(nn.Module):
    def __init__(self, depth=2, bw=10, bh=20, n_hidden=25):
        super().__init__()
        
        self.max_tetrimino = 6
        self.act = nn.ReLU

        # unet like arch
        self.unet_down = nn.Sequential()
        self.unet_up = nn.Sequential()
        self.mid = nn.Sequential()

        for i in range(1, depth):
            # conv + pool + act
            self.unet_down.append(nn.Sequential(
                nn.Conv2d(i**2, (i+1)**2, (2,2), stride=1, padding=1),
                nn.MaxPool2d(2),
                self.act()
            ))

        for i in range(depth, 1, -1):
            # convtransposed + pool + act
            self.unet_up.append(nn.Sequential(
                nn.ConvTranspose2d(i**2, (i-1)**2, (2,2), stride=2, padding=0),
                self.act()
            ))

        if depth:
            self.mid.append(nn.Linear(1, 100))
            self.mid.append(self.act())
            self.mid.append(nn.Linear(100, 1000))
            self.mid.append(self.act())
            self.mid.append(nn.Linear(1000, depth**2))
            self.mid.append(self.act())

        self.pos = nn.Sequential(
            nn.Linear(bw * bh, bw),
            self.act()
        )
        self.rot = nn.Sequential(
            nn.Linear(bw * bh, 4),
            self.act()
        )

        for _ in range(n_hidden):
            self.pos.append(nn.Linear(bw, bw))
            self.pos.append(self.act())
            self.rot.append(nn.Linear(4, 4))
            self.rot.append(self.act())

        self.pos.append(nn.Softmax(dim=0))
        self.rot.append(nn.Softmax(dim=0))

    def forward(self, b, n):
        d = [b.unsqueeze(1)]
        for s in self.unet_down:
            d.append(s(d[-1]))

        mid = self.mid(n / self.max_tetrimino)
        d[-1] = d[-1] + mid.unsqueeze(2).unsqueeze(3).expand(*d[-1].shape)

        # going up with skip connections
        b = d[-1]
        for i, s in enumerate(self.unet_up):
            b = s(b) + d[-i - 2]

        p = self.pos(b.view(-1))
        r = self.rot(b.view(-1))

        return p, r


class TetrisMaster(nn.Module):
	def __init__(self, n_layers=5, bw=10, bh=20, c_hidden=32):
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


