import pytest
from collections import Counter
from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(day="04", examples=[("1", 2)], result=337)
round_2 = dataset_parametrization(day="04", examples=[("2", 3)], result=231)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    assert sum(1 for line in dataset.lines() if len(set(line.split())) == len(line.split())) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    assert sum(1 for line in dataset.lines()
               if len(set(tuple(sorted(Counter(w).items())) for w in line.split())) == len(line.split())) \
           == dataset.result
