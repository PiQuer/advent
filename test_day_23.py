import pytest
from pathlib import Path
import numpy as np
import itertools
import math

from utils import shift


energies = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
home_columns = {'A': 2, 'B': 4, 'C': 6, 'D': 8}
forbidden = tuple(itertools.product((0,), (c for c in home_columns.values())))


def get_data(input_file):
    result = []
    boards = Path(input_file).read_text().split('\n\n')
    for board in boards:
        lines = board.splitlines()
        result.append(np.array([list(line) + [' ']*(len(lines[0])-len(line)) for line in lines]))
    return [r[1:-1, 1:-1] for r in result]


def get_data_part_one(input_file):
    result = get_data(input_file)
    return result[0] if len(result) == 1 else result


def get_data_part_two(input_file):
    boards = get_data(input_file)
    unfold = (" #D#C#B#A# ", " #D#B#A#C# ")
    result = []
    for board in boards:
        if board.shape[0] == 3:
            new_data = np.zeros_like(board, shape=(board.shape[0]+2, board.shape[1]))
            new_data[0:2] = board[0:2]
            new_data[2:4] = np.array([list(line) for line in unfold])
            new_data[4:] = board[4:]
        else:
            new_data = board
        result.append(new_data)
    return result[0] if len(result) == 1 else result


def visualize(board, mask=None):
    if mask is not None:
        board = np.copy(board)
        board[np.logical_and(board != '#', ~mask)] = ' '
    [print(''.join(line)) for line in board]


def one_step(start, board: np.array):
    shifted = np.stack([shift(start, amount=amount, axis=axis, fill=False) for amount in (1, -1) for axis in (0, 1)])
    return np.logical_and.reduce(np.stack([np.logical_or.reduce(shifted), board == '.', ~start]))


def check_blocking(amphipod, board, z1):
    home1 = home_columns[amphipod]
    for b in np.isin(board[0], ('A', 'B', 'C', 'D')).nonzero()[0]:
        home2 = home_columns[board[0, b]]
        if home1 < b < z1 < home2 or home2 < z1 < b < home1:
            return True
    return False


def legal_move(amphipod, board, x0, z0, z1):
    if (z0, z1) in forbidden:
        return False
    if z0 == 0 and check_blocking(amphipod, board, z1):
        return False
    homerun = z1 == home_columns[amphipod] and np.all(board[z0+1:, z1] == amphipod)
    if x0 == 0:
        return homerun
    return z0 == 0 or homerun


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
        if ax0 == 0 or ax1 != home_columns[board[ax0, ax1]] or np.any(board[ax0+1:, ax1] != board[ax0, ax1]):
            index[i] = True
    return amphipods[0][index], amphipods[1][index]


def to_tuple(board):
    return tuple(tuple(d) for d in board)


def is_winning(board):
    for key, value in home_columns.items():
        if np.any(board[1:, value] != key):
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


@pytest.mark.parametrize("input_file,get_data_fn,expected",
                         (("input/day_23_optimize.txt", get_data_part_one, 460),
                          ("input/day_23_example.txt", get_data_part_one, 12521),
                          ("input/day_23.txt", get_data_part_one, 12530),
                          ("input/day_23_example.txt", get_data_part_two, 44169),
                          ("input/day_23.txt", get_data_part_two, 50492)))
def test_day_23(input_file, get_data_fn, expected):
    board = get_data_fn(input_file)
    result = play(board)
    assert result == expected


debug_part_one = [to_tuple(board) for board in get_data_part_one("input/day_23_debug_01.txt")]
debug_part_two = [to_tuple(board) for board in get_data_part_two("input/day_23_debug_02.txt")]
