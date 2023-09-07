import numpy as np
import torch


tetriminos = {
    'L': np.array([
            [0, 0, 1],
            [1, 1, 1]
        ]),
    'J': np.array([
            [1, 1, 1],
            [0, 0, 1]
        ]),
    'S': np.array([
            [0, 1, 1],
            [1, 1, 0]
        ]),
    'Z': np.array([
            [1, 1, 0],
            [0, 1, 1]
        ]),
    'square': np.array([
            [1, 1],
            [1, 1]
        ]),
    'bar': np.array([
            [1, 1, 1, 1]
        ]),
}

# standart reward
s_r = 0.001

rewards = {
    'line_clear_reward': 15 * s_r,
    'height_penalty': -1 * s_r,
    'incorrect_place_penalty': -3 * s_r,
    'lose_game_penalty': -10 * s_r,
}

class Tetris:
    def __init__(self, w=10, h=20):
        self.width = w
        self.height = h

        self.board = np.zeros((self.height, self.width))
        self.buffer = np.zeros((self.height, self.width))
        self.next = 5

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

        # print(col, self.height - np.argmax(self.board[:, col]))
        if h - max(self.board[:, col] * range(self.height)) > self.height:
            return rewards['lose_game_penalty'], rewards['lose_game_penalty']

        max_h = get_height(self.board)
        for i in range(0, self.height - h + 1):
            next_board = self.board.copy()
            try:
                next_board[i:i+h, col:col+w] += cur_tetrimino
            except ValueError:
                return rewards['incorrect_place_penalty'], rewards['incorrect_place_penalty']

            if np.sum(next_board > 1):
                break
            self.buffer = next_board.copy()
        i = self.height - i

        f = rewards['height_penalty'] if (i - h) >= max_h else -rewards['height_penalty']
        r = 10*(max_h / self.height) * rewards['height_penalty']

        return r, f

    def check_complete_lines(self):
        # check complete lines
        r = 0
        i, j, = 0, 0
        lines = np.sum(self.buffer, 1)
        if np.sum(lines == self.width):
            next_board = np.zeros(self.board.shape)
            i, j = 0, 0
            while i < self.height:
                if lines[-i-1] == 0 or abs(j) >= self.height-1:
                    break
                if lines[-i-1] == self.width:
                    j+=1
                    r += 1

                next_board[-i-1] = self.buffer[-j-1]
                i+=1
                j+=1
            self.buffer = next_board
        return r * rewards['line_clear_reward']

    def turn(self, col, rot, update=True):
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
        if update:
            # update board and next piece
            self.board = self.buffer.copy()
            self.next = np.random.randint(0, len(tetriminos))

        return rp, rr

    def print(self):
        print(list(tetriminos.values())[self.next])
        print()
        b = []
        for idx, i in enumerate(self.board):
            s = ' '.join(['@' if j else '.' for j in i]) + ' ' + str(self.height - idx)
            # b.append(s)
            print(s)
        print(' '.join([str(i) for i in range(self.width)]))


def get_height(buffer):
    hs = [i for i in range(buffer.shape[1]) if np.sum(buffer[-i-1])]
    return np.argmax(hs) if len(hs) > 0 else 0

if __name__ == '__main__':
    t = Tetris()

    while True:
        t.set_random_state()
        t.print()
        col, rot = input('input col and rot: ').split(' ')
        r = t.turn(int(col), int(rot))
        print(f'reward: {r}')

