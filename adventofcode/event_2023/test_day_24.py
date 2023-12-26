"""
--- Day 24: Never Tell Me The Odds ---
https://adventofcode.com/2023/day/24
"""
import logging
from dataclasses import dataclass
from functools import partial
from itertools import starmap, combinations

import numpy as np
import pytest
import tinyarray as ta
from more_itertools import quantify, consume

from adventofcode.utils import dataset_parametrization, DataSetBase

YEAR= "2023"
DAY= "24"

@dataclass(frozen=True)
class HailTrajectory:
    pos: ta.ndarray_int
    v: ta.ndarray_int

    def position_at_time(self, t: float) -> ta.ndarray_float:
        return self.pos + t * self.v

    def two_points(self) -> tuple[ta.ndarray_float, ta.ndarray_float]:
        return self.position_at_time(0) / 10**14, self.position_at_time(10**13) / 10**14

    def time_at_position(self, pos: ta.ndarray_float) -> float | None:
        t = np.linalg.norm(pos - self.pos) / np.linalg.norm(self.v)
        if np.allclose(np.array(self.pos) + t * np.array(self.v), np.array(pos)):
            return t
        if np.allclose(np.array(self.pos) - t * np.array(self.v), np.array(pos)):
            return -t
        return None

    def intersect(self, other: "HailTrajectory") -> float | None:
        t = np.linalg.norm(self.pos - other.pos) / np.linalg.norm(other.v - self.v)
        if np.allclose(t * np.array(self.v, dtype=np.float64) + np.array(self.pos, dtype=np.float64),
                  t * np.array(other.v, dtype=np.float64) + np.array(other.pos, dtype=np.float64)):
            return t
        if np.allclose(-t * np.array(self.v, dtype=np.float64) + np.array(self.pos, dtype=np.float64),
                  -t * np.array(other.v, dtype=np.float64) + np.array(other.pos, dtype=np.float64)):
            return -t
        return None


class DataSet(DataSetBase):
    def trajectories(self):
        for line in self.lines():
            pos_str, v_str = line.split(sep='@', maxsplit=1)
            yield HailTrajectory(ta.array((*map(int, pos_str.split(',')),)), ta.array((*map(int, v_str.split(',')),)))

round_1 = dataset_parametrization(
    year=YEAR, day=DAY, examples=[("", (7, 27, 2))], result=(2*10**14, 4*10**14, 15107), dataset_class=DataSet)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[], result=856642398545208, dataset_class=DataSet)


def intersect_x_y(h1: HailTrajectory, h2: HailTrajectory) -> np.ndarray | None:
    a = np.array((h1.v, -h2.v)).T
    try:
        inv_a = np.linalg.inv(a[:2])
    except np.linalg.LinAlgError:
        logging.info("Parallel: %s %s", h1, h2)
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


def validate(stone_trajectory: HailTrajectory, dataset: DataSet) -> bool:
    for trajectory in dataset.trajectories():
        t = stone_trajectory.intersect(trajectory)
        if t is None or t < 0:
            return False
    return True


def plot(trajectories: set[HailTrajectory], stone: HailTrajectory):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    tmax = max(map(stone.intersect, trajectories))
    for trajectory in trajectories:
        t_start = trajectory.position_at_time(0) / 10**14
        t_end = trajectory.position_at_time(tmax) / 10**14
        ax.plot([t_start[0], t_end[0]], [t_start[1], t_end[1]], [t_start[2], t_end[2]])
    stone_start = stone.position_at_time(0) / 10**14
    stone_stop = stone.position_at_time(tmax) / 10**14
    ax.plot([stone_start[0], stone_stop[0]], [stone_start[1], stone_stop[1]], [stone_start[2], stone_stop[2]],
            marker='o')
    plt.show()


def four_points(dataset: DataSet, *args):
    trajectories = list(dataset.trajectories())
    consume((logging.info(trajectories[p].two_points()) for p in args))

@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    trajectories = list(dataset.trajectories())
    four_points(dataset, *(0, 1, 2, 3))
    a = ta.array((386070420374239.1, 249140435425791.9, 215101388198216.3))
    b = ta.array((370019511484545.1, 251239400434818.9, 226275290156335.44))
    t1 = trajectories[3].time_at_position(a)
    t2 = trajectories[2].time_at_position(b)
    v = (b-a) / (t2 - t1)
    p0 = a - t1 * v
    stone = HailTrajectory(ta.array(np.round(p0).astype(np.int64)), ta.array(np.round(v).astype(np.int64)))
    # plot(set(trajectories), stone)
    assert validate(stone, dataset)
    # assert dataset.result == np.sum(stone.pos)
