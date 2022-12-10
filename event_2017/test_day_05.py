from itertools import count
import numpy as np
import pytest
from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(day="05", examples=[("", 5)], result=372139)
round_2 = dataset_parametrization(day="05", examples=[("", 10)], result=29629538)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    maze = dataset.np_array()
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
    maze = dataset.np_array()
    instruction = step = 0
    try:
        for step in count():
            maze[instruction] = (offset := maze[instruction]) + np.sign(2.5 - offset)
            instruction += offset
    except IndexError:
        pass
    assert step == dataset.result
