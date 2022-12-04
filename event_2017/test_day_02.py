from itertools import combinations

import pytest
from pathlib import Path


input_files_round_1 = (
    (Path("input/day_02_example_01.txt"), 18),
    (Path("input/day_02.txt"), 36174),
)


input_files_round_2 = (
    (Path("input/day_02_example_02.txt"), 9),
    (Path("input/day_02.txt"), 244),
)


@pytest.mark.parametrize("input_file,expected", input_files_round_1)
def test_round_1(input_file, expected):
    checksum = 0
    for line in input_file.read_text().splitlines():
        numbers = list(int(c) for c in line.split())
        checksum += max(numbers) - min(numbers)
    assert checksum == expected


@pytest.mark.parametrize("input_file,expected", input_files_round_2)
def test_round_2(input_file, expected):
    checksum = 0
    for line in input_file.read_text().splitlines():
        for a, b in combinations((int(c) for c in line.split()), 2):
            ma, mi = max(a, b), min(a, b)
            if ma % mi == 0:
                checksum += ma // mi
                break
        else:
            assert False
    assert checksum == expected
