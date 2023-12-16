"""
--- Day 1: No Time for a Taxicab ---
https://adventofcode.com/2016/day/1
"""
import numpy as np
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

round_1 = dataset_parametrization(year="2016", day="01", examples=[("_01", 5), ("_02", 2), ("_03", 12)], result=300)
round_2 = dataset_parametrization(year="2016", day="01", examples=[("_04", 4)], result=159)


rotation = {'R': np.array([[0, -1], [1, 0]]), 'L': np.array([[0, 1], [-1, 0]])}


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    data = dataset.text().split(', ')
    pos = np.array([0, 0])
    heading = np.array([1, 0])
    for direction in data:
        heading = np.matmul(rotation[direction[0]], heading)
        pos += heading * int(direction[1:])
    assert np.sum(np.abs(pos)) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    data = dataset.text().split(', ')
    pos = np.array([0, 0])
    heading = np.array([1, 0])
    seen = {tuple(pos)}
    done = False
    for direction in data:
        heading = np.matmul(rotation[direction[0]], heading)
        for v in range(1, int(direction[1:]) + 1):
            pos += heading
            if tuple(pos) in seen:
                done = True
                break
            seen.add(tuple(pos))
        if done:
            break
    assert np.sum(np.abs(pos)) == dataset.result
