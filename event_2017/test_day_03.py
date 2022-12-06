import pytest
import math
from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(day="03", examples=[("1", 2), ("2", 3), ("3", 2), ("4", 31), ("5", 0)], result=371)


def distance(i: int):
    if i == 1:
        return 0
    edge = math.isqrt(i-1)
    edge -= (edge + 1) % 2
    return abs((((i - edge**2 - 1) % (edge+1))+1) - (edge+1)//2) + (edge+1)//2


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    assert distance(int(dataset.text())) == dataset.result
