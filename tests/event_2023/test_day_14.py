from functools import cached_property

import numpy as np
import pytest
import tinyarray as ta
from more_itertools import consume

from utils import dataset_parametrization, DataSetBase

# from utils import generate_rounds

year="2023"
day="14"


class DataSet(DataSetBase):
    @cached_property
    def array(self) -> np.ndarray:
        return self.np_array_bytes()

    def cube_rocks(self) -> dict[ta.ndarray_int, bool]:
        return {ta.array(a): True for a in np.argwhere(self.array == b'#').reshape((-1, 2))}

    def rocks(self) -> list[ta.ndarray_int]:
        yield from map(ta.array, np.argwhere(self.array == b'O').reshape((-1, 2)))

round_1 = dataset_parametrization(year=year, day=day, examples=[("", 136)], result=109385, dataset_class=DataSet)
round_2 = dataset_parametrization(year=year, day=day, examples=[("", 64)], result=None, dataset_class=DataSet)
# pytest_generate_tests = generate_rounds(round_1, round_2)


def roll(cube_rocks: dict[ta.ndarray_int, bool], rocks: dict[ta.ndarray_int], rock: ta.ndarray_int):
    row, column = rock
    while row > 0 and not cube_rocks.get(ta.array((row-1, column)), False) \
            and not rocks.get(ta.array((row-1, rock[1])), False):
        row -= 1
    rocks[ta.array((row, column))] = True


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    cube_rocks = dataset.cube_rocks()
    rocks = {}
    consume(map(lambda r: roll(cube_rocks, rocks, r), dataset.rocks()))
    assert sum(dataset.array.shape[0] - r[0] for r in rocks.keys()) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    pass
