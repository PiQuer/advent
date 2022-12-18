"""
https://adventofcode.com/2022/day/18
"""
import pytest
import numpy as np
from more_itertools import circular_shifts
import numpy_indexed as npi

from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(day="18", examples=[("", 64)], result=3662)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    coordinates = np.loadtxt(dataset.input_file, dtype=int, delimiter=",")
    c_sorted = coordinates[np.lexsort((coordinates[:, 0], coordinates[:, 2], coordinates[:, 1]))]
    diffs = 0
    for c, c1, c2 in circular_shifts((0, 1, 2)):
        g = npi.group_by(c_sorted[:, (c1, c2)])
        diffs += sum(map(np.count_nonzero, map(lambda x: x == 1, map(np.diff, g.split_array_as_list(c_sorted[:, c])))))
    assert len(c_sorted) * 6 - diffs * 2 == dataset.result
