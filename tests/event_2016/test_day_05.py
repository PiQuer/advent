"""
--- Day 5: How About a Nice Game of Chess? ---
https://adventofcode.com/2016/day/5
"""
import hashlib
from itertools import islice
from typing import Iterator

import pytest

from utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def password_chars(self, n=1) -> Iterator[str]:
        pretext = self.text()
        suffix = 0
        while True:
            if (digest := hashlib.md5(f"{pretext}{suffix}".encode()).hexdigest()).startswith("0" * 5):
                yield digest[5:5+n]
            suffix += 1


round_1 = dataset_parametrization("2016", "05", examples=[("", "18f47a30")], result="4543c154", dataset_class=DataSet)
round_2 = dataset_parametrization("2016", "05", examples=[("", "05ace8e3")], result="1050cbbd", dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    assert ''.join(islice(dataset.password_chars(), 0, 8)) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    password = [" "] * 8
    it = dataset.password_chars(n=2)
    while " " in password:
        pos, next_char = next(it)
        if (pos := int(pos, 16)) < 8 and password[pos] == " ":
            password[pos] = next_char
    assert ''.join(password) == dataset.result
