"""
--- Day 3: Spiral Memory ---
https://adventofcode.com/2017/day/3
"""
import math
from collections import defaultdict
from itertools import islice, chain, repeat, count, cycle, dropwhile, accumulate

import numpy as np
import pytest
from more_itertools import repeat_each

from utils import dataset_parametrization, DataSetBase, np_grid

round_1 = dataset_parametrization("2017", "03",
                                  examples=[("1", 2), ("2", 3), ("3", 2), ("4", 31), ("5", 0)], result=371)
round_2 = dataset_parametrization("2017", "03", examples=[], result=369601)


def distance(i: int):
    if i == 1:
        return 0
    edge = math.isqrt(i-1)
    edge -= (edge + 1) % 2
    return abs((((i - edge**2 - 1) % (edge+1))+1) - (edge+1)//2) + (edge+1)//2


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    assert distance(int(dataset.text())) == dataset.result


def spiral(n=None):
    spiral_direction = chain.from_iterable(
        repeat(v, r) for r, v in
        zip(repeat_each(count(1), 2), cycle(np.array(a, dtype=np.int32) for a in ((1, 0), (0, 1), (-1, 0), (0, -1)))))
    spiral_iterator = chain((np.array((0, 0), dtype=np.int32),), spiral_direction)
    return spiral_iterator if n is None else islice(spiral_iterator, n)


@pytest.mark.parametrize(**round_1)
def test_round_1_with_spiral(dataset: DataSetBase):
    assert np.sum(np.abs(sum(spiral(int(dataset.text()))))) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    memory = defaultdict(lambda: 0)
    memory[(0, 0)] = 1
    input_data = int(dataset.text())

    def store_in_memory(pos: np.array):
        value = memory[tuple(pos)] = sum(memory[tuple(pos + a)] for a in np_grid())
        return value
    it = dropwhile(lambda x: x <= input_data, (store_in_memory(pos) for pos in accumulate(spiral())))
    assert next(it) == dataset.result
