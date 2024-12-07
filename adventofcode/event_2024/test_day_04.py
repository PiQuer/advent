"""
--- Day 4: Ceres Search ---
https://adventofcode.com/2024/day/04
"""

import numpy as np
import pytest

YEAR = "2024"
DAY = "04"

from adventofcode.utils import dataset_parametrization, DataSetBase, shift, adjacent_with_diag, diag

part_1 = dataset_parametrization(year=YEAR, day=DAY, part=1, examples=[("", 18)])
part_2 = dataset_parametrization(year=YEAR, day=DAY, part=2, examples=[("", 9)])


@pytest.mark.parametrize(**part_1)
def test_part_1(dataset: DataSetBase):
    result = 0
    for axis_0, axis_1 in adjacent_with_diag():
        b = dataset.np_array_bytes
        f = True
        for letter in (b'X', b'M', b'A', b'S'):
            f = np.logical_and(f, b == letter)
            for amount, axis in ((axis_0, 0), (axis_1, 1)):
                b = shift(b, amount=amount, axis=axis, fill=b'.')
        result += np.sum(f)
    dataset.assert_answer(result)


@pytest.mark.parametrize(**part_2)
def test_part_2(dataset: DataSetBase):
    result = 0
    b = dataset.np_array_bytes
    f = b == b'A'
    for axis_0, axis_1 in diag():
        m = s = b
        for amount, axis in ((axis_0, 0), (axis_1, 1)):
            m = shift(m, amount=amount, axis=axis, fill=b'.')
            s = shift(s, amount=-amount, axis=axis, fill=b'.')
        result += np.logical_and(f, np.logical_and(m == b'M', s == b'S'))
    dataset.assert_answer(np.sum(result == 2))
