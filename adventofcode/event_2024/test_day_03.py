"""
--- Day 3: Mull It Over ---
https://adventofcode.com/2024/day/03
"""

import re
from operator import mul

import pytest

YEAR = "2024"
DAY = "03"

from adventofcode.utils import dataset_parametrization, DataSetBase

round_1 = dataset_parametrization(year=YEAR, day=DAY, part=1, example_results=[161])
round_2 = dataset_parametrization(year=YEAR, day=DAY, part=2, examples=[("2", 48)])


def do_mul(text: str):
    return sum(mul(*map(int, x)) for x in re.findall(r'mul\((\d+),(\d+)\)', text))

@pytest.mark.parametrize(**round_1)
def test_part_1(dataset: DataSetBase):
    dataset.assert_answer(do_mul(dataset.text()))


@pytest.mark.parametrize(**round_2)
def test_part_2(dataset: DataSetBase):
    dataset.assert_answer(
        do_mul(re.sub(r"(?<=don't\(\))(.*?)(?=(do\(\)|$))", '', dataset.text(), flags=re.DOTALL)))
