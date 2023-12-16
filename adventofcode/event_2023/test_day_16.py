"""
--- Day 16: The Floor Will Be Lava ---
https://adventofcode.com/2023/day/16
"""
from dataclasses import dataclass, field

import pytest
import tinyarray as ta

from adventofcode.utils import dataset_parametrization, DataSetBase

# from utils import generate_rounds

YEAR="2023"
DAY= "16"

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", None)], result=None, dataset_class=DataSetBase)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", None)], result=None, dataset_class=DataSetBase)
# pytest_generate_tests = generate_rounds(round_1, round_2)


@dataclass
class Tile:
    coordinates: ta.ndarray_int
    beams: set = field(default_factory=set)

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    pass


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    pass
