import pytest
from pathlib import Path
from utils import dataset_parametrization, DataSetBase

FILE_NAMES = {'example': Path("input/day_02_example.txt"), 'real': Path("input/day_02.txt")}


class DataSet(DataSetBase):
    def lines(self):
        for line in super().lines():
            yield ord(line[0]) - 65, ord(line[2]) - 88


round_1 = dataset_parametrization(day="02", examples=[("", 15)], result=10310, dataset_class=DataSet)
round_2 = dataset_parametrization(day="02", examples=[("", 12)], result=14859, dataset_class=DataSet)


def calculate(dataset: DataSet, second_column_is_outcome: bool):
    score = 0
    for p1, p2 in dataset.lines():
        if second_column_is_outcome:
            p2 = (p1 + p2 - 1) % 3
        score += p2 + 1 + ((p2 - p1 + 1) % 3) * 3
    return score


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    assert calculate(dataset, second_column_is_outcome=False) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert calculate(dataset, second_column_is_outcome=True) == dataset.result
