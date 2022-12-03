import pytest
from pathlib import Path
import re


input_files_round_1 = (
    (Path("input/day_01_example_01.txt"), 3),
    (Path("input/day_01_example_02.txt"), 4),
    (Path("input/day_01_example_03.txt"), 0),
    (Path("input/day_01_example_04.txt"), 9),
    (Path("input/day_01.txt"), 1182),
)


input_files_round_2 = (
    (Path("input/day_01_example_05.txt"), 6),
    (Path("input/day_01_example_06.txt"), 0),
    (Path("input/day_01_example_07.txt"), 4),
    (Path("input/day_01_example_08.txt"), 12),
    (Path("input/day_01_example_09.txt"), 4),
    (Path("input/day_01.txt"), 1152),
)


@pytest.mark.parametrize("input_file,expected", input_files_round_1)
def test_round_01(input_file, expected):
    input_data = input_file.read_text()
    assert sum(int(match.group(1)) for match in re.finditer(r"(?=(\d)\1)", input_data + input_data[0])) == expected


@pytest.mark.parametrize("input_file,expected", input_files_round_2)
def test_round_02(input_file, expected):
    input_data = input_file.read_text()
    mid = len(input_data) >> 1
    input_data += input_data[:mid]
    assert sum(int(match.group(1)) for match in re.finditer(fr"(?=(\d)\d{{{mid-1}}}\1)", input_data)) == expected
