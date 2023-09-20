import numpy as np


class Tetris:
    def __init__(self, w, h, speed=2):
        # game settings
        self.w = w
        self.h = h
        self.done = None
        self.piece_done = None
        self.tick = None
        self.speed = speed

        # view model
        self.board = None
        self.ctp = None
        self.ltp = None
        self.cur = None
        self.rot = None
        self.next = None
        self.score = None
        self.sr = 0.0001
        self.rwd = 0

        self.reset()

    def reset(self):
        self.board = np.zeros((self.h, self.w))
        self.ctp = [0, self.w//2-1]
        self.ltp = [-10, -10]
        self.cur = 1
        self.rot = 1
        self.next = 5
        self.done = False
        self.piece_done = False
        self.tick = 0
        self.score = 0

        return self.render(), self.next 

    def step(self, action):
        rwd = 0
        ct = np.rot90(get_tetr(self.cur), self.rot)

        # update tetrimino position inside
        self.done = self._run_one_tick(action)
        # if h > 10:
        #     rwd -= self.sr

        # Add reward for longer games
        #rwd += self.tick * self.sr * 0.005
        k = np.max(np.argmax(self.board, 0))
        l = np.argmax(self.board[:, self.ctp[1]:self.ctp[1]+ct.shape[1]], 0)
        l = np.where(l > 0, self.h - l, 0)

        h = self.h - k if k > 0 else 0
        ch = np.max(l)
        if ch + ct.shape[0] >= h:
            rwd -= 0#self.sr
        if self.ctp[1] > self.ltp[1]+1 or self.ltp[1]-1 > self.ctp[1]:
            rwd += self.sr

        if self.piece_done:
            # update board inside
            rwd = self._check_lines(self.render())
            # update cur and next tetrimino inside
            self._spawn_piece()

        self.rwd = rwd
        return (self.board, self.next), rwd, self.done

    def _run_one_tick(self, action):
        self.tick += 1
        if self.render() is None:
            return self.board, True
        coords = self.ctp.copy()
        rot = self.rot
        piece_done = self.piece_done

        # action 0 is do nothing
        if action == 1:
            coords[1] -= 1
        if action == 2:
            coords[1] += 1
        if action == 3:
            rot += 1
        if action == 4:
            rot -= 1

        if self.tick % self.speed == 0:
            coords[0] += 1

        b = self.render(coords, rot)
        if b is None:
            # undo player's move
            coords = self.ctp.copy()
            rot = self.rot
            coords[0] += 1
            b = self.render(coords, rot)

        if b is None:
            # means we cant move pieve lower
            coords = self.ctp.copy()
            rot = self.rot
            piece_done = True
            b = self.board

        # update tetrimino's place
        self._update(pd=piece_done, ctp=coords, r=rot)
#         self.print()

        return False

    def _setup_tetris(self):
        self.board[16:20, 1:10] = 1

    def _check_lines(self, board):
        # compute complete line
        r, rwd = 0, 0
        i, j, = 0, 0
        lines = np.sum(board, 1)
        complete_lines = np.nonzero((lines==self.w) * range(self.h))[0]
        r = len(complete_lines)
        for i in complete_lines:
            i+=1
            board[1:i] = board[:i-1]
        if r:
            rwd += (1 << r) * self.sr
            self.score += rwd / (self.sr**2)
        # update board
        self._update(b=board, ltp=self.ctp.copy())
        self.rwd = rwd
        return rwd

    def _update(self, b=None, pd=None, ctp=None, c=None, r=None, n=None, ltp=None):
        if b is not None:
            self.board = b
        if pd is not None:
            self.piece_done = pd
        if ctp:
            self.ctp = ctp
        if c:
            self.cur = c
        if r is not None:
            self.rot = r
        if n:
            self.next = n
        if ltp:
            self.ltp = ltp

    def _spawn_piece(self):
        next_piece = np.random.randint(len(tetriminos.keys()))
        self.ltp = self.ctp.copy()
        self._update(pd=False, ctp=[0,self.w//2-1], c=self.next, r=0, n=next_piece)

    def render(self, coords=None, rot=None):
        x, y = self.ctp if coords is None else coords
        rot = self.rot if rot is None else rot
        cur = np.rot90(get_tetr(self.cur), rot)
        w, h = cur.shape
        board = self.board.copy()

        if not (0 <= x and x+w <= self.h):
            return
        if not (0 <= y and y+h <= self.w):
            return

        try:
            board[x:x+w, y:y+h] += cur
            if (board > 1).any():
#                 print('collision')
                return
        except ValueError:
#             print('value error')
            return
        return board

    def print(self):
#         print(f'{self.board=}')
        print(f'{self.ctp=}')
        print(f'{self.cur=}')
        print(f'{self.rot=}')
        print(f'{self.next=}')
        print(f'{self.done=}')
        print(f'{self.piece_done=}')
#         print(f'{self.tick=}')
#         print(f'{self.score=}')
        print('='*30)
        print()

    def __str__(self):
        out = []
        ttr = get_tetr(self.next)
        out.append(str(ttr)) 
        for _ in range(3 - len(ttr)):
            out.append('')
        out.append(f'places {self.ctp[1]} - {self.ltp[1]}')
        out.append(f'Ticks {self.tick}')
        out.append(f'Score {self.score}')
        out.append(f'Cur reward {self.rwd}')
        b = []
        for idx, i in enumerate(self.render()):
            s = ' '.join(['@' if j else '.' for j in i]) + ' ' + str(self.h - idx)
            out.append(s)
        out.append(' '.join([str(i) for i in range(self.w)]))
        return '\n'.join(out)   


def get_tetr(idx):
    return list(tetriminos.values())[idx]


c, n = 1, 0
tetriminos = {
    'L': np.array([
            [n, n, c],
            [c, c, c]
        ]),
    'J': np.array([
            [c, c, c],
            [n, n, c]
        ]),
    'S': np.array([
            [n, c, c],
            [c, c, n]
        ]),
    'Z': np.array([
            [c, c, n],
            [n, c, c]
        ]),
    'T': np.array([
            [c, c, c],
            [n, c, n]
        ]),
    'O': np.array([
            [c, c],
            [c, c]
        ]),
    'I': np.array([
            [c, c, c, c]
        ]),
}


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument('-c', '--clear-screen', type=bool, default=True)
    p.add_argument('--w', '--width', type=int, default=10)
    p.add_argument('--h', '--height', type=int, default=20)
    a = p.parse_args()

    t = Tetris(a.w, a.h)

    t.reset()
#     t._setup_tetris()
    while True:
        #if a.clear_screen:
        #    os.system('clear')
        print(t)
        act = input('action: ')
        act = int(act) if act != '' else 0
        r = t.step(act)
    #         print(f'reward: {r}')

