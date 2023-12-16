"""
--- Day 11: Dumbo Octopus ---
https://adventofcode.com/2021/day/11
"""
import numpy as np
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

num_steps = 100
window = (slice(1, -1), slice(1, -1))


class DataSet(DataSetBase):
    def get_data(self):
        data = self.np_array_digits()
        result = np.zeros(np.array(data.shape) + 2, dtype=int)
        result[1:-1, 1:-1] = data
        return result


def shift(array: np.array, axes):
    slices = {-1: (2, None), 0: (1, -1), 1: (None, -2)}
    return array[tuple([slice(*slices[a]) for a in axes])]


def flash(data):
    adjacents = np.stack([shift(data > 9, (x, y)) for x in range(-1, 2) for y in range(-1, 2)]).sum(axis=0)
    mask = np.logical_or(data[window] > 9, data[window] == 0)
    data[window] += adjacents * (~mask)
    data[window][mask] = 0
    return data


def calculate_step(data):
    data[window] += 1
    while True:
        new = flash(np.copy(data))
        if np.array_equal(new, data):
            return
        data[...] = new


round_1 = dataset_parametrization("2021", "11", [("", 1656)], result=1665, dataset_class=DataSet)
round_2 = dataset_parametrization("2021", "11", [("", 195)], result=235, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_part_one(dataset: DataSet):
    data = dataset.get_data()
    result = 0
    for _ in range(num_steps):
        calculate_step(data)
        # noinspection PyUnresolvedReferences
        result += (data[window] == 0).sum()
    assert result == dataset.result


@pytest.mark.parametrize(**round_2)
def test_part_two(dataset: DataSet):
    data = dataset.get_data()
    counter = 0
    while data[window].sum() > 0:
        calculate_step(data)
        counter += 1
    assert counter == dataset.result
