import pytest
from pathlib import Path
from itertools import combinations


input_files_round_1 = (
    (Path("input/day_02_example.txt"), 58 + 43),
    (Path("input/day_02.txt"), 1586300),
)


input_files_round_2 = (
    (Path("input/day_02_example.txt"), 34 + 14),
    (Path("input/day_02.txt"), 3737498),
)


def area(p):
    return tuple(a * b for a, b in combinations(p, 2))


def feet_of_ribbon(p):
    distances = sorted(p)
    return distances[0]*distances[1]*distances[2] + 2*sum(distances[:2])


@pytest.mark.parametrize("input_file,expected", input_files_round_1)
def test_round_1(input_file, expected):
    data = input_file.read_text().splitlines()
    paper = 0
    for line in data:
        a = area((int(p) for p in line.split('x')))
        paper += min(a) + 2 * sum(a)
    assert paper == expected


@pytest.mark.parametrize("input_file,expected", input_files_round_2)
def test_round_2(input_file, expected):
    data = input_file.read_text().splitlines()
    ribbon = 0
    for line in data:
        ribbon += feet_of_ribbon((int(p) for p in line.split('x')))
    assert ribbon == expected
