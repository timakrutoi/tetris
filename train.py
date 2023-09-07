import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from tetris import Tetris, rewards
from tetrismaster import TetrisMaster


if __name__ == '__main__':
    w, h = 10, 20
    game_len = 50
    epoch = 100000
    col = 0
    eps = 1e-12
    gamma = 0.79
    rf = 4
    d = 0
    alpha = 0.8

    game = Tetris(w, h)
    # game.clear_board(True)
    game.set_random_state()
    model = TetrisMaster(bw=w, bh=h).double()

    crit = lambda x: -torch.log(F.softmax(x, dim=0).prod() + eps)
    mse = nn.MSELoss()
    optimizer = optim.Adam(TetrisMaster.parameters(model), lr=1e-3)
    best_loss = 99999
    losses_e = []

    it = tqdm(range(0, epoch))
    for e in it:
        losses = []
        rwds = []
        done = False
        for _ in range(game_len):
            optimizer.zero_grad()
            # x = np.concatenate([game.board.reshape(-1), [game.next]])
            x = torch.tensor(game.board, dtype=torch.double).unsqueeze(0)
            y = model(x, torch.tensor(game.next, dtype=torch.double).view(1, 1))#[0]
            y = F.softmax(y, dim=0)
            with torch.no_grad():
                last_col = col
                if False:
                    col = np.random.choice(range(w), p=F.softmax(y[:w], dim=0))
                    rot = np.random.choice(range(4), p=F.softmax(y[w:], dim=0))
                else:
                    col = y[:w].argmax()
                    rot = y[w:].argmax()
                rp, rr = game.turn(col, rot)
                rwds.append(rp + rr)

                target_col = y[:w].clone()
                target_rot = y[w:].clone()
                target_col[col] += rp #+ gamma * torch.max(y[:w]).item()
                target_rot[rot] += rr #+ gamma * torch.max(y[w:]).item()

                target_col[d] -= rp
                # target_rot[0, target_rot[0] == 0] -= rr


                # if rp <= rewards['lose_game_penalty']:
                game.clear_board()
                d = game.set_random_state()
                    # continue
            # print(y)
            loss_col = F.cross_entropy(y[:w], target_col)
            loss_rot = F.cross_entropy(y[w:], target_rot)
            loss = (1 - alpha) * loss_col + alpha * loss_rot
            loss *= 1e7
            loss = 1 / loss
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            output = f'loss: {round(np.mean(losses), rf)} ' +\
                f'| reward: {round(np.mean(rwds), rf)}'
            # print(model.pipeline[0].weight.grad)

            # output = f'loss1 {loss_col.retain_grad()} | loss2 {loss_rot.retain_grad()}'
            it.set_postfix(str=output)
            # exit(-1)

        losses_e.append(np.mean(losses))
        if e % 20:
            best_loss = loss
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
            }, 'checkpoints/test_model'.format(epoch, best_loss))

    print('Done!')

