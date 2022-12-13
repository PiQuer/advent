"""
https://adventofcode.com/2022/day/13
"""
import pytest
from typing import Union
from operator import methodcaller
from itertools import starmap, count
from more_itertools import first_true
from ast import literal_eval
from functools import cmp_to_key
from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(day="13", examples=[("", 13)], result=5852)
round_2 = dataset_parametrization(day="13", examples=[("", 140)], result=24190)


def lge(a: int, b: int) -> int:
    return -1 if a < b else (1 if a > b else 0)


def compare(left: Union[int, list, str], right: Union[int, list, str]) -> int:
    if isinstance(left, str):
        left = literal_eval(left)
    if isinstance(right, str):
        right = literal_eval(right)
    if isinstance(left, int) and isinstance(right, int):
        return lge(left, right)
    if isinstance(left, list) and isinstance(right, list):
        if result := first_true(map(compare, left, right), 0):
            return result
        return lge(len(left), len(right))
    if isinstance(left, int):
        return compare([left], right)
    return compare(left, [right])


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    assert sum(idx for idx, result in
               zip(count(1), starmap(compare, map(methodcaller('split', '\n'), dataset.separated_by_empty_line())))
               if result == -1) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    sorted_list = sorted(["[[2]]", "[[6]]"] + [line for line in dataset.lines() if line], key=cmp_to_key(compare))
    assert (sorted_list.index("[[2]]") + 1) * (sorted_list.index("[[6]]") + 1) == dataset.result
