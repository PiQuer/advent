import pytest
from pathlib import Path
import numpy as np
import itertools
import math

from utils import shift


energies = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
home_columns = {'A': 3, 'B': 5, 'C': 7, 'D': 9}
forbidden = tuple(itertools.product((1,), (c for c in home_columns.values())))


def get_data(input_file):
    lines = Path(input_file).read_text().splitlines()
    data = np.array([list(line) + [' ']*(len(lines[0])-len(line)) for line in lines])
    return data


def visualize(board, mask=None):
    if mask is not None:
        board = np.copy(board)
        board[np.logical_and(board != '#', ~mask)] = ' '
    [print(''.join(line)) for line in board]


def one_step(start, board: np.array):
    shifted = np.stack([shift(start, amount=amount, axis=axis, fill=False) for amount in (1, -1) for axis in (0, 1)])
    return np.logical_and.reduce(np.stack([np.logical_or.reduce(shifted), board == '.', ~start]))


def legal_move(amphipod, board, x0, z0, z1):
    if (z0, z1) in forbidden:
        return False
    homerun = z1 == home_columns[amphipod] and (z0 == 3 or (z0 == 2 and board[3, z1] == amphipod))
    if x0 == 1:
        return homerun
    if x0 in (2, 3):
        return z0 == 1 or homerun


def find_next_moves(x0, x1, seen, board, energy):
    amphipod = board[x0, x1]
    result = []
    steps = 0
    current = np.zeros_like(board, dtype=bool)
    current[x0, x1] = True
    next_steps = current
    while next_steps.any():
        steps += 1
        next_steps = one_step(current, board)
        for z0, z1 in zip(*next_steps.nonzero()):
            if legal_move(amphipod, board, x0, z0, z1):
                new_board = board.copy()
                new_board[z0, z1] = board[x0, x1]
                new_board[x0, x1] = '.'
                new_board_tuple = to_tuple(new_board)
                new_energy = energy + steps*energies[amphipod]
                if new_board_tuple not in seen or seen[new_board_tuple] > new_energy:
                    result.append((new_board, new_energy))
                    seen[new_board_tuple] = new_energy
        current |= next_steps
    return result


def free_amphipods(board):
    amphipods = np.isin(board, ('A', 'B', 'C', 'D')).nonzero()
    index = np.zeros_like(amphipods[0], dtype=bool)
    for i, (ax0, ax1) in enumerate(zip(*amphipods)):
        if ax0 == 1 or ax1 != home_columns[board[ax0, ax1]] or (ax0 == 2 and board[3, ax1] != board[ax0, ax1]):
            index[i] = True
    return amphipods[0][index], amphipods[1][index]


def to_tuple(board):
    return tuple(tuple(d) for d in board)


def is_winning(board):
    for key, value in home_columns.items():
        if board[2, value] != key or board[3, value] != key:
            return False
    return True


def play(board: np.array):
    min_energy = math.inf
    candidate_states = [(board, 0)]
    seen = {to_tuple(board): 0}
    while candidate_states:
        new_candidate_states = []
        for c in candidate_states:
            if is_winning(c[0]):
                min_energy = c[1] if min_energy is None else min(min_energy, c[1])
                continue
            if c[1] > min_energy:
                continue
            to_move = free_amphipods(c[0])
            for x0, x1 in zip(*to_move):
                new_candidate_states.extend(find_next_moves(x0, x1, seen, *c))
        candidate_states = new_candidate_states
    return min_energy


@pytest.mark.parametrize("input_file,expected",
                         (("input/day_23_example.txt", 12521),
                          ("input/day_23.txt", 12530)))
def test_day_23(input_file, expected):
    board = get_data(input_file)
    result = play(board)
    assert result == expected


debug = [to_tuple(get_data(f"input/day_23_debug_0{d}.txt")) for d in range(1, 8)]
