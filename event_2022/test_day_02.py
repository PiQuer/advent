import pytest
from pathlib import Path

FILE_NAMES = {'example': Path("input/day_02_example.txt"), 'real': Path("input/day_02.txt")}


def get_data(data_set: str):
    with FILE_NAMES[data_set].open() as f:
        for line in f:
            yield ord(line[0]) - 65, ord(line[2]) - 88


def calculate(data_set: str, second_column_is_outcome: bool):
    score = 0
    for p1, p2 in get_data(data_set):
        if second_column_is_outcome:
            p2 = (p1 + p2 - 1) % 3
        score += p2 + 1 + ((p2 - p1 + 1) % 3) * 3
    return score


@pytest.mark.parametrize("data_set,second_column_is_outcome,expected",
                         (("example", False, 15), ("real", False, 10310),
                          ("example", True, 12), ("real", True, 14859)))
def test_score(data_set, expected, second_column_is_outcome):
    assert calculate(data_set, second_column_is_outcome) == expected
