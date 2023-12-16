"""
--- Day 25: Full of Hot Air ---
https://adventofcode.com/2022/day/25
"""
from functools import reduce
from itertools import zip_longest

import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

round_1 = dataset_parametrization(year="2022", day="25", examples=[("", "2=-1=0")], result="20-1-11==0-=0112-222")


digit_map = {"=": -2, "-": -1, "0": 0, "1": 1, "2": 2}
reverse_digit_map = {v: k for k, v in digit_map.items()}


def add(n1: list[int], n2: list[int]) -> list[int]:
    carry = (0, 0)
    result = [(carry := divmod((a + b + carry[0]) + 2, 5))[1] - 2 for a, b in zip_longest(n1, n2, fillvalue=0)]
    if carry[0]:
        result.append(carry[0])
    return result


def convert_to_list(line: str) -> list[int]:
    return list(map(digit_map.get, reversed(line)))


def convert_from_list(d: list[int]) -> str:
    return ''.join(map(reverse_digit_map.get, reversed(d)))


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    assert convert_from_list(reduce(add, map(convert_to_list, dataset.lines()))) == dataset.result
