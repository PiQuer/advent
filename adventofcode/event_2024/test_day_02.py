"""
--- Day 2: Red-Nosed Reports ---
https://adventofcode.com/2024/day/02
"""

import numpy as np
import pytest

YEAR= "2024"
DAY= "02"

from adventofcode.utils import dataset_parametrization, DataSetBase

round_1 = dataset_parametrization(year=YEAR, day=DAY, part=1)
round_2 = dataset_parametrization(year=YEAR, day=DAY, part=2)

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    result = 0
    for line in dataset.lines():
        a = np.diff(list(map(int, line.split())))
        result += np.all(np.logical_and(0 < a, a < 4)) or np.all(np.logical_and(-4 < a, a < 0))
    dataset.assert_answer(result)


def condition_1(x):
    return np.logical_and(0 < x, x < 4)


def condition_2(x):
    return np.logical_and(-4 < x, x < 0)


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    result = 0
    for line in dataset.lines():
        a = np.diff(list(map(int, line.split())))
        for condition in (condition_1, condition_2):
            ia = np.argwhere(np.logical_not(condition(a))).reshape(-1).tolist()
            if len(ia) > 2:
                continue
            if (ia == [] or ia == [0] or ia == [a.shape[0]-1]) or \
                    (len(ia) == 1 and \
                        (condition(a[ia[0] - 1] + a[ia[0]]) or
                         condition(a[ia[0]] + a[ia[0] + 1]))) or \
                    (len(ia) == 2 and abs(ia[0] - ia[1]) == 1 and condition(a[ia[0]] + a[ia[1]])):
                result += 1
    dataset.assert_answer(result)
