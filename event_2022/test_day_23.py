"""
https://adventofcode.com/2022/day/23
"""
from itertools import islice, takewhile
from operator import mul
import pytest
import numpy as np
import tinyarray as ta
from collections import deque, defaultdict
from typing import Iterable, Optional
from more_itertools import first, lstrip, consume, quantify, iterate

from utils import dataset_parametrization, DataSetBase


neighbor_directions = {
    ta.array((-1, 0)): (ta.array((-1, -1)), ta.array((-1, 0)), ta.array((-1, 1))),
    ta.array((0, -1)): (ta.array((-1, -1)), ta.array((0, -1)), ta.array((1, -1))),
    ta.array((1, 0)): (ta.array((1, -1)), ta.array((1, 0)), ta.array((1, 1))),
    ta.array((0, 1)): (ta.array((-1, 1)), ta.array((0, 1)), ta.array((1, 1)))
}


def minmax(it: Iterable[ta.array]) -> tuple[ta.array, ta.array]:
    it = iter(it)
    result_min = np.array(next(it), dtype=int)
    result_max = result_min.copy()
    for a in it:
        result_min = np.where(result_min > a, a, result_min)
        result_max = np.where(result_max < a, a, result_max)
    return ta.array(result_min), ta.array(result_max + 1)


def visualize(board: set[ta.array], b_min: Optional[ta.array] = None):
    _b_min, b_max = minmax(board)
    b_min = _b_min if b_min is None else b_min
    pane = np.full(tuple(b_max - b_min), '.')
    pane[tuple(zip(*(b - b_min for b in board)))] = '#'
    consume(map(print, (''.join(row) for row in pane)))


def neighbors(pos: ta.array, board: set[ta.array]) -> dict[str, int]:
    return {n: sum(1 for d in neighbor_directions[n] if (pos + d) in board) for n in neighbor_directions}


class DataSet(DataSetBase):
    def board(self) -> set[ta.array]:
        return set(map(ta.array, np.argwhere(self.np_array_bytes() == b'#')))


round_1 = dataset_parametrization(day="23", examples=[("1", 110)], result=4049, dataset_class=DataSet)
round_2 = dataset_parametrization(day="23", examples=[("1", 20)], result=1021, dataset_class=DataSet)


def move(board: set[ta.array], try_list: deque[ta.array]) -> bool:
    proposed_moves = defaultdict(lambda: [])
    changed = False
    for elve in board:
        nm = neighbors(elve, board)
        if 0 < quantify(nm.values()) < 4:
            proposed_moves[elve + first(lstrip(try_list, lambda t: nm[t]))].append(elve)
    for pm in (_ for _ in proposed_moves if len(proposed_moves[_]) == 1):
        board.remove(proposed_moves[pm][0])
        board.add(pm)
        changed = True
    try_list.append(try_list.popleft())
    return changed


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    try_list = deque((ta.array((-1, 0)), ta.array((1, 0)), ta.array((0, -1)), ta.array((0, 1))))
    board = dataset.board()
    consume(islice(takewhile(bool, iterate(lambda x: move(board, try_list), True)), 10+1))
    min_b, max_b = minmax(board)
    assert mul(*(max_b - min_b)) - len(board) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    try_list = deque((ta.array((-1, 0)), ta.array((1, 0)), ta.array((0, -1)), ta.array((0, 1))))
    board = dataset.board()
    assert quantify(takewhile(bool, iterate(lambda x: move(board, try_list), True))) == dataset.result
