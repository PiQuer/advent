"""
--- Day 01: Historian Hysteria ---
https://adventofcode.com/2024/day/01
"""
from functools import cache

import numpy as np
import pytest

YEAR= "2024"
DAY= "01"

from adventofcode.utils import dataset_parametrization, DataSetBase

round_1 = dataset_parametrization(year=YEAR, day=DAY, part=1)
round_2 = dataset_parametrization(year=YEAR, day=DAY, part=2, example_results=[31])


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    sorted = np.sort(dataset.np_array(), axis=0)
    dataset.assert_answer(np.sum(np.abs(sorted[:,0] - sorted[:, 1])))


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    a = dataset.np_array()
    @cache
    def score(input: int) -> int:
        return input * np.sum(a[:, 1] == input)

    dataset.assert_answer(sum(map(score, a[:, 0])))
