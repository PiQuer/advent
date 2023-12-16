"""
--- Day 2: Corruption Checksum ---
https://adventofcode.com/2017/day/2
"""
from itertools import combinations

import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

round_1 = dataset_parametrization("2017", "02", [("_01", 18)], result=36174)
round_2 = dataset_parametrization("2017", "02", [("_02", 9)], result=244)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    checksum = 0
    for line in dataset.lines():
        numbers = list(int(c) for c in line.split())
        checksum += max(numbers) - min(numbers)
    assert checksum == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    checksum = 0
    for line in dataset.lines():
        for a, b in combinations((int(c) for c in line.split()), 2):
            ma, mi = max(a, b), min(a, b)
            if ma % mi == 0:
                checksum += ma // mi
                break
        else:
            assert False
    assert checksum == dataset.result
