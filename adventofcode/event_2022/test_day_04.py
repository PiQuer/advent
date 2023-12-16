"""
--- Day 4: Camp Cleanup ---
https://adventofcode.com/2022/day/4
"""
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def lines(self):
        for line in super().lines():
            range1, range2 = line.split(',')
            yield tuple(int(i) for i in range1.split('-')), tuple(int(i) for i in range2.split('-'))


round_1 = dataset_parametrization("2022", "04", examples=[("", 2)], result=534, dataset_class=DataSet)
round_2 = dataset_parametrization("2022", "04", examples=[("", 4)], result=841, dataset_class=DataSet)


def contained(pair1: tuple, pair2: tuple):
    return pair1[0] >= pair2[0] and pair1[1] <= pair2[1]


def contained_both_ways(pair1: tuple, pair2: tuple):
    return contained(pair1=pair1, pair2=pair2) or contained(pair1=pair2, pair2=pair1)


def overlap(pair1: tuple, pair2: tuple):
    return pair1[0] <= pair2[0] <= pair1[1] or pair2[0] <= pair1[0] <= pair2[1]


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    assert sum(contained_both_ways(pair1, pair2) for pair1, pair2 in dataset.lines()) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    assert sum(overlap(pair1, pair2) for pair1, pair2 in dataset.lines()) == dataset.result
