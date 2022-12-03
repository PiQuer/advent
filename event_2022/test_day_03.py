import pytest
from pathlib import Path


input_files_round_1 = (
    (Path("input/day_03_example.txt"), 157),
    (Path("input/day_03.txt"), 7568),
)

input_files_round_2 = (
    (Path("input/day_03_example.txt"), 70),
    (Path("input/day_03.txt"), 2780),
)


def priority(c):
    return ord(c) - 38 if c < 'a' else ord(c) - 96


@pytest.mark.parametrize("data_file,expected", input_files_round_1)
def test_round_1(data_file, expected):
    priority_sum = 0
    for line in data_file.read_text().split('\n'):
        half = len(line) >> 1
        priority_sum += priority((set(line[:half]) & set(line[half:])).pop())
    assert priority_sum == expected


@pytest.mark.parametrize("data_file,expected", input_files_round_2)
def test_round_2(data_file, expected):
    priority_sum = 0
    data = data_file.read_text().split('\n')
    for g in range(0, len(data), 3):
        priority_sum += priority((set(data[g]) & set(data[g+1]) & set(data[g+2])).pop())
    assert priority_sum == expected
