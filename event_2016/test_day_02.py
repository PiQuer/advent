import pytest
from pathlib import Path
import numpy as np


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


input_files_round_1 = (
    (Path("input/day_02_example.txt"), 1985),
    (Path("input/day_02.txt"), 78293),
)


input_files_round_2 = (
    (Path("input/day_02_example.txt"), 0x5DB3),
    (Path("input/day_02.txt"), 0xAC8C8),
)


def calculate_code(data, pos, keypad, base=10):
    code = 0
    for line in data:
        for move in line:
            new_pos = np.array(pos) + directions[move]
            pos = new_pos if keypad[tuple(new_pos)] else pos
        code = code * base + keypad[tuple(pos)]
    return code


@pytest.mark.parametrize("input_file,expected", input_files_round_1)
def test_round_1(input_file, expected):
    assert calculate_code(input_file.read_text().splitlines(), (2, 2), keypad_1) == expected


@pytest.mark.parametrize("input_file,expected", input_files_round_2)
def test_round_2(input_file, expected):
    assert calculate_code(input_file.read_text().splitlines(), (3, 1), keypad_2, base=16) == expected
