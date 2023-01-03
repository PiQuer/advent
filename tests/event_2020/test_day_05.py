"""
https://adventofcode.com/2020/day/5
"""
import pytest
from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(year="2020", day="05", examples=[("", 820)], result=None)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    pass
