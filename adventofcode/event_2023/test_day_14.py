"""
--- Day 14: Parabolic Reflector Dish ---
https://adventofcode.com/2023/day/14
"""
from functools import cache
from itertools import count

import numpy as np
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

YEAR= "2023"
DAY= "14"


round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 136)], dataset_class=DataSetBase, part=1)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 64)], dataset_class=DataSetBase, part=2)

@cache
def roll_col(col: tuple[bytes, ...]) -> bytearray:
    stopper = 0
    result = bytearray(b'.' * len(col))
    for i, b in enumerate(col):
        if b == b'O':
            result[stopper] = ord(b'O')
            stopper += 1
        elif b == b'#':
            result[i] = ord(b'#')
            stopper = i+1
    return result


def roll(rocks: np.ndarray) -> None:
    for index, col in enumerate(rocks.T):
        rocks[:, index] = np.frombuffer(roll_col(tuple(col)), dtype=rocks.dtype)


def total_load(rocks: np.ndarray) -> int:
    rock_coordinates = np.argwhere(np.flip(rocks, axis=0) == b'O')
    return sum(rock_coordinates[:, 0]) + len(rock_coordinates)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    rocks = dataset.np_array_bytes
    roll(rocks)
    dataset.assert_answer(total_load(rocks))


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    rocks = dataset.np_array_bytes
    seen = {(h := rocks.tobytes()): 0}
    cycle = [rocks.copy()]
    index = 0
    for index in count(1):
        for _ in range(4):
            roll(rocks)
            rocks = np.rot90(rocks, k=-1)
        if (h := rocks.tobytes()) in seen:
            break
        seen[h] = index
        cycle.append(rocks.copy())
    final = cycle[(1000000000 - index) % (index - seen[h]) + seen[h]]
    dataset.assert_answer(total_load(final))
