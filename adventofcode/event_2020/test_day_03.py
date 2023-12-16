"""
--- Day 3: Toboggan Trajectory ---
https://adventofcode.com/2020/day/3
"""
from functools import reduce
from itertools import count
from operator import mul

import numpy as np
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

round_1 = dataset_parametrization(year="2020", day="03", examples=[("", 7)], result=292)
round_2 = dataset_parametrization(year="2020", day="03", examples=[("", 336)], result=9354744432)


def slope(it, dataset: DataSetBase) -> int:
    a = dataset.np_array_bytes()
    rows, cols = a.shape
    return reduce(
        mul, (np.sum(a[tuple(zip(*zip(range(0, rows, down), (x % cols for x in count(0, right)))))] == b'#')
              for right, down in it), 1)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset):
    it = ((3, 1),)
    assert slope(it, dataset) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset):
    it = ((1, 1), (3, 1), (5, 1), (7, 1), (1, 2))
    assert slope(it, dataset) == dataset.result
