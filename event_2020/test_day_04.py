import pytest
from dataclasses import dataclass
import re
from utils import dataset_parametrization, DataSetBase
from more_itertools import peekable


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
        if not self.valid_year(self.byr, 1920, 2002):
            return False
        if not self.valid_year(self.iyr, 2010, 2020):
            return False
        if not self.valid_year(self.eyr, 2020, 2030):
            return False
        if (m := re.match(r"(\d+)(in|cm)$", self.hgt)) is None:
            return False
        if m.group(2) == "in" and not (59 <= int(m.group(1)) <= 76):
            return False
        if m.group(2) == "cm" and not (150 <= int(m.group(1)) <= 193):
            return False
        if re.match(r"#[\da-f]{6}$", self.hcl) is None:
            return False
        if self.ecl not in ("amb", "blu", "brn", "gry", "grn", "hzl", "oth"):
            return False
        if re.match(r"\d{9}$", self.pid) is None:
            return False
        return True


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


round_1 = dataset_parametrization(day="04", examples=[("1", 2)], result=190, dataset_class=DataSet)
round_2 = dataset_parametrization(day="04", examples=[("2", 0), ("3", 4)], result=121, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    assert sum(1 for _ in dataset.valid_passports()) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert sum(1 for _ in dataset.valid_passports(input_validation=True)) == dataset.result
