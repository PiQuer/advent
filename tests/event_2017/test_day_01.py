import pytest
import re

from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization("2017", "01", [("_01", 3), ("_02", 4), ("_03", 0), ("_04", 9)], result=1182)
round_2 = dataset_parametrization("2017", "01", [("_05", 6), ("_06", 0), ("_07", 4), ("_08", 12), ("_09", 4)],
                                  result=1152)


@pytest.mark.parametrize(**round_1)
def test_round_01(dataset: DataSetBase):
    input_data = dataset.text()
    assert sum(int(match.group(1)) for match in re.finditer(r"(?=(\d)\1)", input_data + input_data[0])) == \
           dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_02(dataset: DataSetBase):
    input_data = dataset.text()
    mid = len(input_data) >> 1
    input_data += input_data[:mid]
    assert sum(int(match.group(1)) for match in re.finditer(fr"(?=(\d)\d{{{mid-1}}}\1)", input_data)) == \
           dataset.result
