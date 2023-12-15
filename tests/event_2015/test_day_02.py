"""
--- Day 2: I Was Told There Would Be No Math ---
https://adventofcode.com/2015/day/2
"""
from itertools import combinations

import pytest

from utils import dataset_parametrization, DataSetBase

round_1 = dataset_parametrization("2015", "02", [("", 58 + 43)], result=1586300)
round_2 = dataset_parametrization("2015", "02", [("", 34 + 14)], result=3737498)


def area(p):
    return tuple(a * b for a, b in combinations(p, 2))


def feet_of_ribbon(p):
    distances = sorted(p)
    return distances[0]*distances[1]*distances[2] + 2*sum(distances[:2])


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    data = dataset.lines()
    paper = 0
    for line in data:
        a = area((int(p) for p in line.split('x')))
        paper += min(a) + 2 * sum(a)
    assert paper == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    data = dataset.lines()
    ribbon = 0
    for line in data:
        ribbon += feet_of_ribbon((int(p) for p in line.split('x')))
    assert ribbon == dataset.result
