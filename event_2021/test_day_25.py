import pytest
import numpy as np
from pathlib import Path


def get_data(input_file):
    data = Path(input_file).read_text().splitlines()
    return np.array([list(line) for line in data])


def visualize(board):
    [print(''.join(line)) for line in board]


def move(board):
    more = False
    for axis, symbol in ((1, '>'), (0, 'v')):
        to_move = (board == symbol) & np.roll((board == '.'), shift=-1, axis=axis)
        more |= np.any(to_move)
        board[to_move] = '.'
        board[np.roll(to_move, shift=1, axis=axis)] = symbol
    return more


@pytest.mark.parametrize("input_file,expected",
                         (("input/day_25_example.txt", 58),
                          ("input/day_25.txt", 532)))
def test_day_25(input_file, expected):
    board = get_data(input_file)
    result = 1
    while move(board):
        result += 1
    assert result == expected
