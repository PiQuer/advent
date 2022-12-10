import pytest
import re
from utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def nice_lines_1(self):
        for line in self.lines():
            if re.search(r"([aeiou].*){3}", line) and re.search(r"(.)\1", line) and \
                    (re.search(r"ab|cd|pq|xy", line) is None):
                yield line

    def nice_lines_2(self):
        for line in self.lines():
            if re.search(r"(..).*\1", line) and re.search(r"(.).\1", line):
                yield line


round_1 = dataset_parametrization(day="05", examples=[("1", 2)], result=236, dataset_class=DataSet)
round_2 = dataset_parametrization(day="05", examples=[("2", 2)], result=51, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    assert sum(1 for _ in dataset.nice_lines_1()) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert sum(1 for _ in dataset.nice_lines_2()) == dataset.result
