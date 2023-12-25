"""
--- Day 24: Never Tell Me The Odds ---
https://adventofcode.com/2023/day/24
"""
from dataclasses import dataclass
from functools import partial
from itertools import starmap, combinations

import numpy as np
import pytest
import tinyarray as ta
from more_itertools import quantify

from adventofcode.utils import dataset_parametrization, DataSetBase

YEAR= "2023"
DAY= "24"

@dataclass(frozen=True)
class HailTrajectory:
    pos: ta.ndarray_int
    v: ta.ndarray_int

    def position_at_time(self, t: float) -> ta.ndarray_float:
        return self.pos + t * self.v

class DataSet(DataSetBase):
    def trajectories(self):
        for line in self.lines():
            pos_str, v_str = line.split(sep='@', maxsplit=1)
            yield HailTrajectory(ta.array((*map(int, pos_str.split(',')),)), ta.array((*map(int, v_str.split(',')),)))

round_1 = dataset_parametrization(
    year=YEAR, day=DAY, examples=[("", (7, 27, 2))], result=(2*10**14, 4*10**14, 15107), dataset_class=DataSet)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", None)], result=None, dataset_class=DataSet)


def intersect_x_y(h1: HailTrajectory, h2: HailTrajectory) -> np.ndarray | None:
    a = np.array((h1.v, -h2.v)).T
    try:
        inv_a = np.linalg.inv(a[:2])
    except np.linalg.LinAlgError:
        return None
    return inv_a@(np.array(h2.pos-h1.pos)[:2])


def intersect_2d_testarea(h1: HailTrajectory, h2: HailTrajectory, minimum: int, maximum: int) -> bool:
    solution = intersect_x_y(h1, h2)
    if solution is not None:
        if np.all(solution >= 0):
            pos = h1.position_at_time(solution[0])
            return minimum <= pos[0] <= maximum and minimum <= pos[1] <= maximum
        return False
    # Trajectories are parallel, but could still have common points inside the test area
    # do a few tests to rule out the edge condition of the paths being equal or having zero velocity
    assert h1.v[0] != 0 or h1.v[1] != 0
    idx = 0 if h1.v[0] != 0 else 1
    t = (h2.pos - h1.pos)[idx] / h1.v[idx]
    assert h1.pos + t * h1.v != h2.pos
    return False

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    minimum, maximum, result = dataset.result
    assert quantify(starmap(partial(intersect_2d_testarea, minimum=minimum, maximum=maximum),
                            combinations(dataset.trajectories(), 2))) == result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert dataset.result is None
