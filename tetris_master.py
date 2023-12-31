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
        return nn.functional.adaptive_avg_pool2d(x, 1).reshape((b, c))


class TetrisMaster2(nn.Module):
    def __init__(self, depth, hidden_num, hidden_size):
        super().__init__()

        self.max_tetrimino = 7
        self.num_actions = 4
        self.hidden_size = hidden_size
        self.act = nn.Tanh

        self.conv = nn.Sequential()

        for i in range(depth):
            self.conv.append(nn.Conv2d(2**i, 2**(i+1), kernel_size=3, stride=1, padding=1))
            self.conv.append(self.act())

        self.conv.append(GlobalPool2d())        

        self.mid = nn.Sequential(
            nn.Linear(self.max_tetrimino, 32),
            self.act(),
            nn.Linear(32, 128),
            self.act(),
            nn.Linear(128, 2**depth),
            self.act()
        )

        self.out = nn.Sequential(
            nn.Linear(2**depth, self.hidden_size),
            self.act()
        )

        for _ in range(hidden_num):
            self.out.append(
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.LayerNorm(self.hidden_size),
                    self.act(),
                )
            )

        self.end = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_actions),
            self.act(),
            nn.Softmax(dim=0)
        )

    def forward(self, b, n):
        b = b - 0.5
        if len(b.shape) == 2:
            b = b.unsqueeze(0)
        x = self.conv(b.unsqueeze(0))

        oh = F.one_hot(n.squeeze(0).squeeze(0).long(), self.max_tetrimino)
        r = self.mid(oh.double())

        x = (x + r).view(-1)

        x = self.out[:2](x)
        for l in self.out[1:]:
            x = l(x) + x

        x = self.end(x)

        return x


if __name__ == '__main__':
    model = TetrisMaster2()
    print(model)

    y = model(torch.rand((10, 20)), torch.tensor([1]).view(-1, 1))
    print(y)
    y.sum().backward()

