"""
--- Day 3: Squares With Three Sides ---
https://adventofcode.com/2016/day/3
"""
import numpy as np
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

round_1 = dataset_parametrization("2016", "03", [], result=983)
round_2 = dataset_parametrization("2016", "03", [], result=1836)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    data = np.sort(dataset.np_array(), axis=1)
    assert np.sum(data[:, 0] + data[:, 1] > data[:, 2]) == 983


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    data = np.sort(dataset.np_array().transpose().reshape(-1, 3), axis=1)
    assert np.sum(data[:, 0] + data[:, 1] > data[:, 2]) == 1836
