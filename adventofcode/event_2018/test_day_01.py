"""
--- Day 1: Chronal Calibration ---
https://adventofcode.com/2018/day/1
"""
from itertools import cycle

import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

round_1 = dataset_parametrization("2018", "01", [], result=490)
round_2 = dataset_parametrization("2018", "01", [], result=70357)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    result = 0
    for line in dataset.lines():
        c = int(line[1:])
        result += c if line[0] == "+" else -c
    assert result == 490


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    result = 0
    seen = {result}
    for line in cycle(dataset.lines()):
        c = int(line[1:])
        result += c if line[0] == "+" else -c
        if result in seen:
            break
        seen.add(result)
    assert result == 70357
