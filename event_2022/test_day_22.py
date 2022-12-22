"""
https://adventofcode.com/2022/day/22
"""
from functools import partial
from itertools import takewhile, islice, cycle
from operator import ne
import pytest
import numpy as np
import re
import tinyarray as ta
from typing import Iterator, Optional
from more_itertools import consume, last, lstrip

from utils import dataset_parametrization, DataSetBase


def visualize(c: np.array, pos: Optional[ta.array] = None):
    to_visualize = c[0]
    if pos is not None:
        to_visualize = to_visualize.copy()
        to_visualize[tuple(pos)] = 2
    to_visualize = np.vectorize(inv_byte_map.get)(to_visualize)
    consume(map(print, (''.join(row) for row in to_visualize)))


byte_map = {' ': -1, '#': 0, '.': 1, 'P': 2}
inv_byte_map = {v: k for k, v in byte_map.items()}
inv_facing_map = {'>': '<', '<': '>', 'v': '^', '^': 'v'}


def rotate(m: np.array, direction: str, pos: ta.array, facing: int) -> tuple[np.array, ta.array, int]:
    rotated = m
    if direction == "R":
        pos = ta.array((m.shape[2] - 1 - pos[1], pos[0]))
        rotated = np.rot90(m, axes=(1, 2))
        facing += 1
    elif direction == "L":
        pos = ta.array((pos[1], m.shape[1] - 1 - pos[0]))
        rotated = np.rot90(m, k=-1, axes=(1, 2))
        facing -= 1
    assert rotated.base is m or rotated.base is m.base or rotated is m
    return rotated, pos, facing % 4


class DataSet(DataSetBase):
    def board(self) -> tuple[np.array, ta.array]:
        _map = self.separated_by_empty_line()[0].split("\n")
        _max = max(map(len, _map))
        x = np.array([list(map(byte_map.get, row + " "*(_max - len(row)))) for row in _map], dtype=int)
        x = np.concatenate((x[np.newaxis, :], np.indices(x.shape)))
        return x, ta.array((0, next(np.nditer(np.argwhere(x[0, 0] == 1)))))

    def instructions(self) -> Iterator[tuple[str, int]]:
        i_str = "S" + self.separated_by_empty_line()[1]
        return map(lambda m: (m.group(1), int(m.group(2))), re.finditer(r"([SRL])(\d+)", i_str))


def walk(m: np.array, pos: ta.array, instr: tuple[str, int], facing: int) -> tuple[np.array, ta.array, int]:
    m, pos, facing = rotate(m, instr[0], pos, facing)
    row = m[0, pos[0]]
    indices = np.arange(len(row))
    start = indices[pos[1]]
    pos = ta.array((pos[0],
                    last(takewhile(row.__getitem__, islice(lstrip(cycle(np.nditer(indices[np.where(row >= 0)])),
                                                                  partial(ne, start)), instr[1] + 1)))))
    return m, pos, facing


round_1 = dataset_parametrization(day="22", examples=[("", 6032)], result=30552, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    board, pos = dataset.board()
    facing = 0
    for instr in dataset.instructions():
        board, pos, facing = walk(board, pos, instr, facing)
    assert 1000 * (pos[0] + 1) + 4 * (pos[1] + 1) + 0 == dataset.result
