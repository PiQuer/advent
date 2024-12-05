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

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    result = 0
    for line in dataset.lines():
        a = np.diff(list(map(int, line.split())))
        result += np.all(np.logical_and(0 < a, a < 4)) or np.all(np.logical_and(-4 < a, a < 0))
    dataset.assert_answer(result)
