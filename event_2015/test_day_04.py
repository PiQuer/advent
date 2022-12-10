import pytest
from hashlib import md5
from itertools import count

from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(day="04", examples=[("1", 609043), ("2", 1048970)], result=282749)
round_2 = dataset_parametrization(day="04", examples=[], result=9962624)


class Day04:
    prefix: str

    def test_round_1(self, dataset: DataSetBase):
        prefix = dataset.text()
        assert next(i for i in count() if md5(f"{prefix}{i}".encode()).hexdigest().startswith(self.prefix)) \
               == dataset.result


@pytest.mark.parametrize(**round_1)
class TestRound1(Day04):
    prefix = '0' * 5


@pytest.mark.parametrize(**round_2)
class TestRound2(Day04):
    prefix = '0' * 6
