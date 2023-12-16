"""
--- Day 2: Password Philosophy ---
https://adventofcode.com/2020/day/2
"""
from dataclasses import dataclass
from typing import Iterator

import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase


@dataclass
class PasswordLine:
    m1: int
    m2: int
    char: str
    pwd: str


class DataSet(DataSetBase):
    def lines(self) -> Iterator[PasswordLine]:
        for line in super().lines():
            r, c, p = line.split(' ')
            m1, m2 = (int(i) for i in r.split('-'))
            yield PasswordLine(m1=m1, m2=m2, char=c[0], pwd=p)


round_1 = dataset_parametrization(year="2020", day="02", examples=[("", 2)], result=638, dataset_class=DataSet)
round_2 = dataset_parametrization(year="2020", day="02", examples=[], result=699, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round1(dataset: DataSet):
    assert sum(1 for p in dataset.lines() if p.m1 <= p.pwd.count(p.char) <= p.m2) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round2(dataset: DataSet):
    assert sum(1 for p in dataset.lines()
               if (p.pwd[p.m1 - 1] == p.char) + (p.pwd[p.m2 - 1] == p.char) == 1) == dataset.result
