"""
--- Day 6: Guard Gallivant ---
https://adventofcode.com/2024/day/06
"""
import numpy as np
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

# from adventofcode.utils import generate_parts

YEAR= "2024"
DAY= "06"

class DataSet(DataSetBase):
    def start(self):
        return np.argwhere(self.np_array_bytes == b'^')


part_1 = dataset_parametrization(year=YEAR, day=DAY, part=1, dataset_class=DataSet)
part_2 = dataset_parametrization(year=YEAR, day=DAY, part=2, dataset_class=DataSet)


def get_route(m: np.array, start: np.array) -> bool:
    position = start.copy()
    while True:
        path = m[:position[0], position[1]]
        obstacle = np.argwhere(path == b'#').reshape(-1)
        if len(obstacle):
            path[obstacle[-1] + 1:position[0]] = b'X'
            position[0] = m.shape[0] - 1 - position[1]
            position[1] = obstacle[-1] + 1
            m = np.rot90(m)
        else:
            path[:] = b'X'
            return False


@pytest.mark.parametrize(**part_1)
def test_part_1(dataset: DataSet):
    m = dataset.np_array_bytes
    position = np.argwhere(m == b'^')[0]
    m[*position] = b'X'
    get_route(m, position)
    dataset.assert_answer(np.sum(m == b'X'))


@pytest.mark.parametrize(**part_2)
def test_part_2(dataset: DataSet):
    m = dataset.np_array_bytes
    start = np.argwhere(m == b'^')[0]
    possible_locations = np.argwhere(m == b'.')
    result = 0
    for location in possible_locations:
        position = start.copy()
        t = m.copy()

