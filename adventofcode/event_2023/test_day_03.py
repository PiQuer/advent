"""
--- Day 3: Gear Ratios ---
https://adventofcode.com/2023/day/3
"""
from collections import defaultdict

import numpy as np
import pytest
import tinyarray as ta
from more_itertools import peekable

from adventofcode.utils import dataset_parametrization, DataSetBase, inbounds

round_1 = dataset_parametrization(year="2023", day="03", examples=[("", 4361)], part=1)
round_2 = dataset_parametrization(year="2023", day="03", examples=[("", 467835)], part=2)


def is_symbol(a: np.array, idx: ta.array, symbol: bytes = None) -> bool:
    if symbol is None:
        return inbounds(ta.array(a.shape), idx) and (not a[*idx].isdigit()) and a[*idx] != b'.'
    return inbounds(ta.array(a.shape), idx) and a[*idx] == symbol


def get_number(a: np.array, index_iter: peekable) -> int:
    idx = ta.array(next(index_iter))
    if not a[*idx].isdigit():
        return 0
    result = 0
    start = idx
    use = any(is_symbol(a, start + ta.array(offset)) for offset in ((-1, -1), (0, -1), (1, -1)))
    while a[*idx].isdigit():
        result = 10 * result + int(a[*idx])
        use = use or any(is_symbol(a, idx + ta.array(offset)) for offset in ((-1, 0), (1, 0)))
        if idx[1] == a.shape[1] - 1:
            return result if use else 0
        idx = ta.array(next(index_iter))
    use = use or any(is_symbol(a, idx + ta.array(offset)) for offset in ((-1, 0), (0, 0), (1, 0)))
    return result if use else 0


def update_gear(a: np.array, index_iter: peekable, gears: defaultdict):
    idx = ta.array(next(index_iter))
    if not a[*idx].isdigit():
        return
    number = 0
    start = idx
    stars = set()
    def extend_stars(s: set, i):
        s.update(set(i + ta.array(offset) for offset in ((-1, 0), (0, 0), (1, 0))
                     if is_symbol(a, i + ta.array(offset), b'*')))
    extend_stars(stars, start + ta.array((0, -1)))
    while a[*idx].isdigit():
        number = 10 * number + int(a[*idx])
        extend_stars(stars, idx)
        if idx[1] == a.shape[1] - 1:
            break
        idx = ta.array(next(index_iter))
    extend_stars(stars, idx)
    for star in stars:
        gears[star].add(number)
    return


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    result = 0
    a = dataset.np_array_bytes
    index_iter = peekable(np.ndindex(a.shape))
    while index_iter:
        result += get_number(a, index_iter)
    dataset.assert_answer(result)


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    a = dataset.np_array_bytes
    index_iter = peekable(np.ndindex(a.shape))
    gears = defaultdict(set)
    while index_iter:
        update_gear(a, index_iter, gears)
    dataset.assert_answer(sum(np.prod(list(v)) for k, v in gears.items() if len(v) == 2))
