"""
--- Day 5: A Maze of Twisty Trampolines, All Alike ---
https://adventofcode.com/2017/day/5
"""
from itertools import count

import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

round_1 = dataset_parametrization(year="2017", day="05", examples=[("", 5)], result=372139)
round_2 = dataset_parametrization(year="2017", day="05", examples=[("", 10)], result=29629538)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    maze = [int(i) for i in dataset.lines()]
    instruction = step = 0
    try:
        for step in count():
            maze[instruction] += 1
            instruction += maze[instruction] - 1
    except IndexError:
        pass
    assert step == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    maze = [int(i) for i in dataset.lines()]
    instruction = step = 0
    try:
        for step in count():
            maze[instruction] = (offset := maze[instruction]) + (1 if offset < 3 else -1)
            instruction += offset
    except IndexError:
        pass
    assert step == dataset.result
