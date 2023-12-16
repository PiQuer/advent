"""
--- Day 17: Trick Shot ---
https://adventofcode.com/2021/day/17
"""
import math

import numpy as np
import pytest
from numpy.lib import recfunctions as rfn

from adventofcode.utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def get_data(self):
        return rfn.structured_to_unstructured(
            np.fromregex(self.input_file, r"target area: x=(\d+)\.\.(\d+), y=(-\d+)\.\.(-\d+)",
                         dtype="int32,int32,int32,int32")[0])


round_1 = dataset_parametrization(year="2021", day="17", examples=[("", 45)], result=9180, dataset_class=DataSet)
round_2 = dataset_parametrization(year="2021", day="17", examples=[("", 112)], result=3767, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_part_one(dataset: DataSet):
    data = dataset.get_data()
    n = -data[2]
    assert n*(n-1)/2 == dataset.result


def trajectory_hit(px, py, data):
    x, y = 0, 0
    while x <= data[1] and y >= data[2]:
        x += px
        y += py
        if data[0] <= x <= data[1] and data[2] <= y <= data[3]:
            return 1
        py -= 1
        px = max(0, px-1)
    return 0


@pytest.mark.parametrize(**round_2)
def test_part_two(dataset: DataSet):
    data = dataset.get_data()
    x1, x2, y1, y2 = data
    px_min = math.ceil(-1/2 + math.sqrt((1/2)**2 + 2*x1))
    px_max = x2
    py_max = -y1-1
    py_min = y1
    result = 0
    for px in range(px_min, px_max+1):
        for py in range(py_min, py_max+1):
            result += trajectory_hit(px, py, data)
    assert result == dataset.result
