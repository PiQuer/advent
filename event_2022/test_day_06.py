import pytest
from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(day="06", examples=[("1", 7), ("2", 5), ("3", 6), ("4", 10), ("5", 11)],
                                  result=1794)
round_2 = dataset_parametrization(day="06", examples=[("1", 19), ("2", 23), ("3", 23), ("4", 29), ("5", 26)],
                                  result=2851)


def start_of(input_string: str, num_unique: int):
    return next(k for k in range(num_unique, len(input_string))
                if len(set(input_string[k-num_unique:k])) == num_unique)


@pytest.mark.parametrize(**round_1)
def test_round1(dataset: DataSetBase):
    assert start_of(dataset.text(), 4) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round2(dataset: DataSetBase):
    assert start_of(dataset.text(), 14) == dataset.result
