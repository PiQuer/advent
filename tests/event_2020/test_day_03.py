"""
--- Day 3: Toboggan Trajectory ---
https://adventofcode.com/2020/day/3
"""
from functools import reduce
from itertools import count
from operator import mul
from typing import Iterator

import numpy as np
import pytest

from utils import dataset_parametrization, DataSetBase

round_1 = dataset_parametrization(year="2020", day="03", examples=[("", 7)], result=292)
round_2 = dataset_parametrization(year="2020", day="03", examples=[("", 336)], result=9354744432)


class Day03:
    it: Iterator[tuple[int, int]]

    def test_slope(self, dataset: DataSetBase):
        a = dataset.np_array_bytes()
        rows, cols = a.shape
        assert reduce(
            mul, (np.sum(a[tuple(zip(*zip(range(0, rows, down), (x % cols for x in count(0, right)))))] == b'#')
                  for right, down in self.it), 1) == dataset.result


@pytest.mark.parametrize(**round_1)
class TestRound1(Day03):
    it = ((3, 1),)


@pytest.mark.parametrize(**round_2)
class TestRound2(Day03):
    it = ((1, 1), (3, 1), (5, 1), (7, 1), (1, 2))
