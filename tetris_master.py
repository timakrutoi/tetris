import numpy as numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class TetrisMaster2(nn.Module):
    def __init__(self, depth=2, bw=10, bh=20, n_hidden=256):
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

        self.out = nn.Sequential(
            nn.Linear(bw * bh, 100),
            self.act(),
        )

        for _ in range(n_hidden):
            self.out.append(nn.Linear(100, 100))
            self.out.append(self.act())

        self.out.append(nn.Linear(100, 4))
        self.out.append(nn.Softmax(dim=0))

    def forward(self, b, n):
        d = [b.unsqueeze(0).unsqueeze(0)]
        # print(d[-1].shape)
        for s in self.unet_down:
            d.append(s(d[-1]))

        mid = self.mid(n / self.max_tetrimino)
        mid = mid.unsqueeze(2).unsqueeze(3)
        
        #print(f'{d[-1].shape} + {mid.shape}')
        d[-1] = d[-1] + mid.expand(*d[-1].shape)

        # going up with skip connections
        b = d[-1]
        for i, s in enumerate(self.unet_up):
            b = s(b) + d[-i - 2]

        b = b.view(-1)
        o = self.out(b)

        return o


if __name__ == '__main__':
    model = TetrisMaster2()
    print(model)

    y = model(torch.rand((10, 20)), torch.tensor([1]).view(-1, 1))
    print(y)
    y.sum().backward()

