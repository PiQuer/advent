import pytest
from pathlib import Path
from pytest_cases import parametrize, fixture
from utils import dataset_cases


cases_round_1 = dataset_cases("04", [("", 2)], 534)


input_files_round_1 = (
    (Path("input/day_04_example.txt"), 2),
    (Path("input/day_04.txt"), 534),
)


input_files_round_2 = (
    (Path("input/day_04_example.txt"), 4),
    (Path("input/day_04.txt"), 841),
)


def contained(pair1: tuple, pair2: tuple):
    return pair1[0] >= pair2[0] and pair1[1] <= pair2[1]


def range_to_pair(range1: str, range2: str):
    return tuple(int(i) for i in range1.split('-')), tuple(int(i) for i in range2.split('-'))


def contained_both_ways(range1: str, range2: str):
    pair1, pair2 = range_to_pair(range1, range2)
    return contained(pair1, pair2) or contained(pair2, pair1)


def overlap(range1: str, range2: str):
    pair1, pair2 = range_to_pair(range1, range2)
    return pair1[0] <= pair2[0] <= pair1[1] or pair2[0] <= pair1[0] <= pair2[1]


@pytest.mark.parametrize("input_file,expected", input_files_round_1)
def test_round_1(input_file, expected):
    data = input_file.read_text().splitlines()
    assert sum(contained_both_ways(*line.split(',')) for line in data) == expected


@pytest.mark.parametrize("input_file,expected", input_files_round_2)
def test_round_2(input_file, expected):
    data = input_file.read_text().splitlines()
    assert sum(overlap(*line.split(',')) for line in data) == expected
