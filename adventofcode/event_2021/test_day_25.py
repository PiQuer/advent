"""
--- Day 25: Sea Cucumber ---
https://adventofcode.com/2021/day/25
"""
import numpy as np
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase


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


round_1 = dataset_parametrization("2021", "25", [("", 58)], result=532)


@pytest.mark.parametrize(**round_1)
def test_day_25(dataset: DataSetBase):
    board = dataset.np_array_str()
    result = 1
    while move(board):
        result += 1
    assert result == dataset.result
