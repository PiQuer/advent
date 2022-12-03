import pytest
from pathlib import Path


input_files_round_1 = (
    (Path("input/day_01_example_01.txt"), 0),
    (Path("input/day_01_example_02.txt"), 3),
    (Path("input/day_01_example_03.txt"), 3),
    (Path("input/day_01_example_04.txt"), -1),
    (Path("input/day_01_example_05.txt"), -3),
    (Path("input/day_01.txt"), 0),
)


input_files_round_2 = (
    (Path("input/day_01_example_06.txt"), 5),
    (Path("input/day_01.txt"), 1771),
)


@pytest.mark.parametrize("input_file,expected", input_files_round_1)
def test_floor(input_file, expected):
    data = input_file.read_text()
    assert data.count('(') - data.count(')') == expected


@pytest.mark.parametrize("input_file,expected", input_files_round_2)
def test_basement(input_file, expected):
    data = input_file.read_text()
    floor = 0
    for i, c in enumerate(data):
        floor += 1 if c == '(' else -1
        if floor == -1:
            break
    else:
        i = -1
    assert i+1 == expected
