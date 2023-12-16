"""
--- Day XX: XXXXX ---
https://adventofcode.com/2023/day/XX
"""
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

# from utils import generate_rounds

YEAR= "2023"
DAY= "00"

class DataSet(DataSetBase):
    pass

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", None)], result=None, dataset_class=DataSet)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", None)], result=None, dataset_class=DataSet)
# pytest_generate_tests = generate_rounds(round_1, round_2)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    pass


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    pass
