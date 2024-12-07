"""
--- Day XX: XXXXX ---
https://adventofcode.com/2024/day/XX
"""
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

# from adventofcode.utils import generate_parts

YEAR= "2024"
DAY= "00"

class DataSet(DataSetBase):
    pass

part_1 = dataset_parametrization(year=YEAR, day=DAY, part=1)
part_2 = dataset_parametrization(year=YEAR, day=DAY, part=2)
# part_1 = dataset_parametrization(year=YEAR, day=DAY, part=1, examples=[("", None)], result=None, dataset_class=DataSet)
# part_2 = dataset_parametrization(year=YEAR, day=DAY, part=2, examples=[("", None)], result=None, dataset_class=DataSet)
# pytest_generate_tests = generate_parts(part_1, part_2)


@pytest.mark.parametrize(**part_1)
def test_part_1(dataset: DataSet):
    assert dataset.result is None


@pytest.mark.parametrize(**part_2)
def test_part_2(dataset: DataSet):
    assert dataset.result is None
