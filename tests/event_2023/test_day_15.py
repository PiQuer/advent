from dataclasses import dataclass
from functools import reduce

import pytest

from utils import dataset_parametrization, DataSetBase

# from utils import generate_rounds

year="2023"
day="15"

class DataSet(DataSetBase):
    pass

round_1 = dataset_parametrization(year=year, day=day, examples=[("", 1320)], result=507291, dataset_class=DataSet)
round_2 = dataset_parametrization(year=year, day=day, examples=[("", 145)], result=None, dataset_class=DataSet)
# pytest_generate_tests = generate_rounds(round_1, round_2)


def update(current: int, char: int) -> int:
    return ((current + char) * 17) % 256


def calculate_hash(b: bytes) -> int:
    return reduce(update, bytearray(b), 0)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    assert sum(map(calculate_hash, dataset.bytes().split(b','))) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    pass
