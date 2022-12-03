import pytest
from pathlib import Path
import numpy as np


input_files_round_1 = (
    (Path("input/day_01_example_01.txt"), 5),
    (Path("input/day_01_example_02.txt"), 2),
    (Path("input/day_01_example_03.txt"), 12),
    (Path("input/day_01.txt"), 300),
)


input_files_round_2 = (
    (Path("input/day_01_example_04.txt"), 4),
    (Path("input/day_01.txt"), 159),
)


rotation = {'R': np.array([[0, -1], [1, 0]]), 'L': np.array([[0, 1], [-1, 0]])}


@pytest.mark.parametrize("data_file,expected", input_files_round_1)
def test_round_1(data_file, expected):
    data = data_file.read_text().split(', ')
    pos = np.array([0, 0])
    heading = np.array([1, 0])
    for direction in data:
        heading = np.matmul(rotation[direction[0]], heading)
        pos += heading * int(direction[1:])
    assert expected == np.sum(np.abs(pos))


@pytest.mark.parametrize("data_file,expected", input_files_round_2)
def test_round_2(data_file, expected):
    data = data_file.read_text().split(', ')
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
    assert expected == np.sum(np.abs(pos))
