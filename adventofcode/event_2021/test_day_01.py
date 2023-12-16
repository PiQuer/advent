"""
--- Day 1: Sonar Sweep ---
https://adventofcode.com/2021/day/1
"""
import numpy as np
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(year="2021", day="01", examples=[("", 7)], result=1448)
round_2 = dataset_parametrization(year="2021", day="01", examples=[("", 5)], result=1471)


def count_increasing(data):
    return (data[1:] - data[:-1] > 0).sum()


@pytest.mark.parametrize(**round_1)
def test_part_one(dataset: DataSetBase):
    assert count_increasing(dataset.np_array()) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_part_two(dataset):
    data = np.lib.stride_tricks.sliding_window_view(np.genfromtxt(dataset.input_file, dtype=int), window_shape=(3, ))
    assert count_increasing(data.sum(axis=1)) == dataset.result
