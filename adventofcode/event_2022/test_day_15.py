"""
--- Day 15: Beacon Exclusion Zone ---
https://adventofcode.com/2022/day/15
"""
import re
from functools import partial
from itertools import starmap
from typing import Optional

import pytest
import tinyarray as ta
from more_itertools import consume, pairwise

from adventofcode.utils import dataset_parametrization, DataSetBase


def i_len(interval: list[int]) -> int:
    return interval[1] - interval[0]


def intersect(a: list[int], b: list[int]):
    b[0] = max(b[0], a[1])
    b[1] = max(b[1], a[1])


class EndOfProblem(Exception):
    pass


class DataSet(DataSetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensors = []
        self.beacons = []
        self.get_sensors()

    def get_sensors(self):
        for line in self.lines():
            match = re.search(r"Sensor at x=(-?\d+), y=(-?\d+): closest beacon is at x=(-?\d+), y=(-?\d+)", line)
            self.sensors.append(ta.array(tuple(map(int, match.groups()[:2]))))
            self.beacons.append(ta.array(tuple(map(int, match.groups()[2:4]))))

    @staticmethod
    def get_interval(sensor: ta.array, beacon: ta.array, y: int) -> Optional[list[int]]:
        if (dx := sum(ta.abs(beacon - sensor)) - abs(sensor[1] - y)) >= 0:
            return [sensor[0] - dx, sensor[0] + dx + 1, y]
        return None

    def intervals(self, y: int):
        return filter(bool, starmap(partial(self.get_interval, y=y), zip(self.sensors, self.beacons)))


round_1 = dataset_parametrization(year="2022", day="15",
                                  examples=[("", 26, {'y': 10})], result=4811413, dataset_class=DataSet,
                                  y=2_000_000)
round_2 = dataset_parametrization(year="2022", day="15",
                                  examples=[("", 56000011, {'bound': 20})], result=13171855019123,
                                  dataset_class=DataSet, bound=4_000_000)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    y = dataset.params["y"]
    intervals = sorted(dataset.intervals(y=y), key=lambda x: x[0])
    consume(starmap(intersect, pairwise(intervals)))
    assert sum(map(i_len, intervals)) - len({b for b in dataset.beacons if b[1] == y}) == dataset.result


def intersect_and_check(a: list[int], b: list[int], bound: int):
    b[0] = max(b[0], a[1])
    b[1] = max(b[1], a[1])
    if b[0] - a[1] == 1 and 0 <= a[1] <= bound:
        raise EndOfProblem(a)


def key_fn(x: list[int]):
    return x[0]


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    bound = dataset.params["bound"]
    y = 0
    try:
        consume(map(consume, map(partial(starmap, partial(intersect_and_check, bound=bound)),
                                 map(pairwise, map(partial(sorted, key=key_fn),
                                                   map(dataset.intervals, range(bound, -1, -1)))))))
    except EndOfProblem as e:
        print(f"{e.args[0]} {y}")
        assert e.args[0][1] * 4_000_000 + e.args[0][2] == dataset.result
    else:
        assert False
