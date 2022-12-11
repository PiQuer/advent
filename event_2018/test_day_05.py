import pytest
import string
from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(day="05", examples=[("", 10)], result=10888)
round_2 = dataset_parametrization(day="05", examples=[("", 4)], result=6952)


def p_reduce(p: list[str], i: int) -> int:
    if i >= len(p) - 1:
        return i
    if p[i] != p[i+1] and p[i].lower() == p[i+1].lower():
        del(p[i:i+2])
        return max(0, i-1)
    return i + 1


def full_reduce(p: list[str]):
    i = 0
    while i < len(p) - 1:
        i = p_reduce(p, i)
    return p


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    p = list(dataset.text())
    assert len(full_reduce(p)) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    p = list(dataset.text())
    assert min(len(full_reduce(pr)) for pr in ([e for e in p if e.lower() != s] for s in string.ascii_lowercase)) \
        == dataset.result
