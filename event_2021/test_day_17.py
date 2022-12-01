import pytest
from pytest_cases import fixture
import numpy as np
from numpy.lib import recfunctions as rfn
import math


@fixture(scope="module", unpack_into="data,expected_1,expected_2")
@pytest.mark.parametrize("input_file,expected_1,expected_2",
                         (("input/day_17_example.txt", 45, 112),
                          ("input/day_17.txt", 9180, 3767)))
def get_data(input_file: str, expected_1, expected_2):
    return rfn.structured_to_unstructured(
        np.fromregex(input_file, r"target area: x=(\d+)\.\.(\d+), y=(-\d+)\.\.(-\d+)",
                     dtype="int32,int32,int32,int32")[0]), \
           expected_1, expected_2


def test_part_one(data, expected_1):
    n = -data[2]
    result = n*(n-1)/2
    assert result == expected_1


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


def test_part_two(data, expected_2):
    x1, x2, y1, y2 = data
    px_min = math.ceil(-1/2 + math.sqrt((1/2)**2 + 2*x1))
    px_max = x2
    py_max = -y1-1
    py_min = y1
    result = 0
    for px in range(px_min, px_max+1):
        for py in range(py_min, py_max+1):
            result += trajectory_hit(px, py, data)
    assert result == expected_2
