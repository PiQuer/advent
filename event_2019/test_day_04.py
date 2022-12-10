from typing import Optional, Iterator
import pytest
import re
from more_itertools import difference, first_true
from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(day="04", examples=[], result=1099)
round_2 = dataset_parametrization(day="04", examples=[], result=710)


def first_non_rising(s: str) -> Optional[int]:
    return first_true(enumerate(difference(int(i) for i in s)), default=(None,), pred=lambda x: x[1] < 0)[0]


def contains_double(s: str, extended=False) -> bool:
    it = re.finditer(r"(.)\1", s)
    if extended:
        return any(m for m in it if m.group(1)*3 not in s)
    return any(it)


def validator(i: int, extended: bool = False) -> tuple[bool, Optional[int]]:
    str_i = str(i)
    fnr = first_non_rising(str_i)
    return (fnr is None and contains_double(str(i), extended=extended)), fnr


def next_valid(i: int, extended: bool = False) -> int:
    valid, fnr = validator(i, extended=extended)
    if valid:
        return i
    if fnr is None:
        return next_valid(i+1, extended=extended)
    str_i = str(i)
    return next_valid(int(str_i[:fnr] + str_i[fnr-1]*(len(str_i) - fnr)), extended=extended)


def valid_generator(lower: int, upper: int, extended: bool = False) -> Iterator[int]:
    current = lower - 1
    while True:
        current = next_valid(current + 1, extended=extended)
        if current > upper:
            break
        yield current


@pytest.mark.parametrize("i,valid",
                         ((111111, True),
                          (223450, False),
                          (123789, False)))
def test_validator(i, valid):
    assert validator(i)[0] == valid


@pytest.mark.parametrize("i, valid",
                         ((112233, True),
                          (123444, False),
                          (111122, True)))
def test_validator_extended(i, valid):
    assert validator(i, extended=True)[0] == valid


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    lower, upper = (int(i) for i in dataset.text().split('-'))
    assert sum(1 for _ in valid_generator(lower, upper)) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    lower, upper = (int(i) for i in dataset.text().split('-'))
    assert sum(1 for _ in valid_generator(lower, upper, extended=True)) == dataset.result
