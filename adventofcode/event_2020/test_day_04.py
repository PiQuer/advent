"""
--- Day 4: Passport Processing ---
https://adventofcode.com/2020/day/4
"""
import re
from dataclasses import dataclass

import pytest
from more_itertools import peekable

from adventofcode.utils import dataset_parametrization, DataSetBase


# pylint: disable=too-many-instance-attributes
@dataclass
class Passport:
    byr: str
    iyr: str
    eyr: str
    hgt: str
    hcl: str
    ecl: str
    pid: str
    cid: str = ""

    @staticmethod
    def valid_year(year: str, lower: int, upper: int) -> bool:
        try:
            return lower <= int(year) <= upper
        except ValueError:
            return False

    def valid(self) -> bool:
        result = True
        if not self.valid_year(self.byr, 1920, 2002):
            result = False
        elif not self.valid_year(self.iyr, 2010, 2020):
            result = False
        elif not self.valid_year(self.eyr, 2020, 2030):
            result = False
        elif (m := re.match(r"(\d+)(in|cm)$", self.hgt)) is None:
            result = False
        elif m.group(2) == "in" and not 59 <= int(m.group(1)) <= 76:
            result = False
        elif m.group(2) == "cm" and not 150 <= int(m.group(1)) <= 193:
            result = False
        elif re.match(r"#[\da-f]{6}$", self.hcl) is None:
            result = False
        elif self.ecl not in ("amb", "blu", "brn", "gry", "grn", "hzl", "oth"):
            result = False
        elif re.match(r"\d{9}$", self.pid) is None:
            result = False
        return result


class DataSet(DataSetBase):
    def valid_passports(self, input_validation=False):
        kwargs = {}
        it = peekable(self.lines())
        for line in it:
            kwargs.update(dict((s[:3], s[4:]) for s in line.split()))
            if line == "" or it.peek(None) is None:
                try:
                    p = Passport(**kwargs)
                except TypeError:
                    pass
                else:
                    if not input_validation or p.valid():
                        yield p
                kwargs = {}


round_1 = dataset_parametrization("2020", "04", examples=[("1", 2)], result=190, dataset_class=DataSet)
round_2 = dataset_parametrization("2020", "04", examples=[("2", 0), ("3", 4)], result=121, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    assert sum(1 for _ in dataset.valid_passports()) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert sum(1 for _ in dataset.valid_passports(input_validation=True)) == dataset.result
