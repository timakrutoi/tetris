import numpy as np
import torch


# colored and not colored
c = 1
n = 0

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
    'square': np.array([
            [c, c],
            [c, c]
        ]),
    'bar': np.array([
            [c, c, c, c]
        ]),
}

# standart reward
s_r = 0.0001

rewards = {
    'line_clear_reward': 30 * s_r,
    'hight_reward': 1 * s_r,
    'success_place_reward': 1 * s_r,
    'repeate_pen': -1 * s_r,
    'incorrect_place_penalty': -1 * s_r,
    'lose_game_penalty': -300 * s_r,
}

class Tetris:
    def __init__(self, w=10, h=20, rewards=rewards):
        self.width = w
        self.height = h

        self.board = np.zeros((self.height, self.width))
        self.buffer = np.zeros((self.height, self.width))
        self.next = 5
        
        self.last_turn = None
        self.rewards = rewards
        self.total_lines_burnt = 0

    def clear_board(self, make_ft=False):
        self.board = np.zeros((self.height, self.width))
        self.buffer = np.zeros((self.height, self.width))       
        if make_ft:
            self.turn(0, 0)

    def set_random_state(self, n_cols=1):
        dots = np.random.randint(0, self.width, size=n_cols)
        for i in range(n_cols):
            # self.board[-i-1] = np.random.randint(0, 2, size=self.width)
            self.board[-i-1] = 1
            self.board[-i-1, dots[i]] = 0

        self.buffer = self.board.copy()
        return dots

    def place_piece(self, col, cur_tetrimino):
        # move to col
        # move down
        h, w = cur_tetrimino.shape

        max_h = get_height(self.board)
        prev_board = self.board.copy()
        for i in range(h, self.height + 1):
            next_board = self.board.copy()

#             print(get_height(self.board) + h, self.height)
            if get_height(self.board) + h > self.height:
                return self.rewards['lose_game_penalty'], self.rewards['lose_game_penalty']

            try:
                i -= h
                next_board[i:i+h, col:col+w] += cur_tetrimino
            except ValueError:
                if i < h:
                    return self.rewards['incorrect_place_penalty'], self.rewards['incorrect_place_penalty']
                else:
                    break
            if np.sum(next_board > 1):
                break
            prev_board = next_board.copy()
        self.board = prev_board.copy()

        r, f = 0, 0

#         print(self.height - i + 1, i-1, h, max_h)
        if (self.height - i + 1) <= max_h:
            r += self.rewards['hight_reward']
            f += self.rewards['hight_reward']

        r += self.rewards['success_place_reward']
        f += self.rewards['success_place_reward']

        return r, f

    def check_complete_lines(self):
        # check complete lines
        r = 0
        i, j, = 0, 0
        lines = np.sum(self.board, 1)
        if np.sum(lines == self.width):
            next_board = np.zeros(self.board.shape)
            i, j = 0, 0
            while i < self.height:
                if lines[-i-1] == 0 or abs(j) >= self.height-1:
                    break
                if lines[-i-1] == self.width:
                    j+=1
                    r+=1

                next_board[-i-1] = self.board[-j-1]
                i+=1
                j+=1
            self.board = next_board
        self.total_lines_burnt += r
#         print(r)
        return r**2 * self.rewards['line_clear_reward']

    def turn(self, col, rot):
        rp, rr = 0, 0
        # rotate piece
        cur_tetrimino = np.rot90(list(tetriminos.values())[self.next], rot)

        r = self.place_piece(col, cur_tetrimino)
        rp += r[0]
        rr += r[1]
        # print(f'1 {rp, rr=}')
        r = self.check_complete_lines()
        rp += r
        rr += r
        # print(f'2 {rp, rr=}')
        self.next = np.random.randint(0, len(tetriminos))

        if self.last_turn is not None:
            rp += self.rewards['repeate_pen'] if self.last_turn[0] == col else 0
            rr += self.rewards['repeate_pen'] if self.last_turn[1] == rot else 0
        self.last_turn = (col, rot)

        return rp, rr

    def print(self):
        print(list(tetriminos.values())[self.next])
        print(f'{self.total_lines_burnt=}')
        print()
        b = []
        for idx, i in enumerate(self.board):
            s = ' '.join(['@' if j else '.' for j in i]) + ' ' + str(self.height - idx)
            # b.append(s)
            print(s)
        print(' '.join([str(i) for i in range(self.width)]))


def get_height(buffer):
    hs = [(np.argmax(buffer[:,i])) for i in range(buffer.shape[1])]
    hs = np.array(hs)
    if np.sum(hs) == 0:
        return 0
    hs[hs==0] = buffer.shape[0]
    return buffer.shape[0] - np.min(hs)

if __name__ == '__main__':
    t = Tetris()

    t.set_random_state()
    while True:
        t.print()
        col, rot = input('input col and rot: ').split(' ')
        r = t.turn(int(col), int(rot))
        print(f'reward: {r}')

