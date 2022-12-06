import pytest
import numpy as np
from utils import dataset_parametrization, DataSetBase


keypad_1 = np.array(
    [[0, 0, 0, 0, 0],
     [0, 1, 2, 3, 0],
     [0, 4, 5, 6, 0],
     [0, 7, 8, 9, 0],
     [0, 0, 0, 0, 0]]
)


keypad_2 = np.array(
    [[0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 2, 3, 4, 0, 0],
     [0, 5, 6, 7, 8, 9, 0],
     [0, 0, 10, 11, 12, 0, 0],
     [0, 0, 0, 13, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0]]
)


directions = {
    'U': np.array((-1, 0)),
    'D': np.array((1, 0)),
    'L': np.array((0, -1)),
    'R': np.array((0, 1))
}


round_1 = dataset_parametrization(day="02", examples=[("", 1985)], result=78293)
round_2 = dataset_parametrization(day="02", examples=[("", 0x5DB3)], result=0xAC8C8)


def calculate_code(data, pos, keypad, base=10):
    code = 0
    for line in data:
        for move in line:
            new_pos = np.array(pos) + directions[move]
            pos = new_pos if keypad[tuple(new_pos)] else pos
        code = code * base + keypad[tuple(pos)]
    return code


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    assert calculate_code(dataset.lines(), (2, 2), keypad_1) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    assert calculate_code(dataset.lines(), (3, 1), keypad_2, base=16) == dataset.result
