import pytest

from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization("2015", "01", [("_01", 0), ("_02", 3), ("_03", 3), ("_04", -1), ("_05", -3)],
                                  result=138)
round_2 = dataset_parametrization("2015", "01", [("_06", 5)], result=1771)


@pytest.mark.parametrize(**round_1)
def test_floor(dataset: DataSetBase):
    data = dataset.text()
    assert data.count('(') - data.count(')') == dataset.result


@pytest.mark.parametrize(**round_2)
def test_basement(dataset: DataSetBase):
    data = dataset.text()
    floor = 0
    for i, c in enumerate(data):
        floor += 1 if c == '(' else -1
        if floor == -1:
            break
    else:
        i = -1
    assert i+1 == dataset.result
