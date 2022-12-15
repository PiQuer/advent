"""
https://adventofcode.com/2022/day/15
"""
from functools import partial
from typing import Optional
import pytest
from more_itertools import consume
from dataclasses import dataclass
import tinyarray as ta
import re
from itertools import pairwise, starmap
from operator import methodcaller
from utils import dataset_parametrization, DataSetBase

@dataclass
class Interval:
    left: int
    right: int
    y: int

    def len(self) -> int:
        return self.right - self.left

def intersect(a: Interval, b: Interval):
    b.left = max(b.left, a.right)
    b.right = max(b.right, a.right)

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
    def get_interval(sensor: ta.array, beacon: ta.array, y: int) -> Optional[Interval]:
        if (dx := sum(ta.abs(beacon - sensor)) - abs(sensor[1] - y)) >= 0:
            return Interval(sensor[0] - dx, sensor[0] + dx + 1, y)

    def intervals(self, y: int):
        return filter(bool, starmap(partial(self.get_interval, y=y), zip(self.sensors, self.beacons)))


round_1 = dataset_parametrization(day="15", examples=[("", 26)], result=4811413, dataset_class=DataSet)
round_2 = dataset_parametrization(day="15", examples=[("", 56000011)], result=13171855019123, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(request, dataset: DataSet):
    y = 10 if "example" in request.node.name else 2_000_000
    intervals = sorted(dataset.intervals(y=y), key=lambda x: x.left)
    consume(starmap(intersect, pairwise(intervals)))
    assert sum(map(methodcaller("len"), intervals)) - len({b for b in dataset.beacons if b[1] == y}) == dataset.result


def intersect_and_check(a: Interval, b: Interval, bound: int):
    b.left = max(b.left, a.right)
    b.right = max(b.right, a.right)
    if b.left - a.right == 1 and 0 <= a.right <= bound:
        raise EndOfProblem(a)

def key_fn(x: Interval):
    return x.left

@pytest.mark.parametrize(**round_2)
def test_round_2(request, dataset: DataSet):
    bound = 20 if "example" in request.node.name else 4_000_000
    y = 0

    try:
        consume(map(consume, map(partial(starmap, partial(intersect_and_check, bound=bound)),
                                 map(pairwise, map(partial(sorted, key=key_fn),
                                                   map(dataset.intervals, range(bound, -1, -1)))))))
    except EndOfProblem as e:
        print(f"{e.args[0]} {y}")
        assert e.args[0].right * 4_000_000 + e.args[0].y == dataset.result
    else:
        assert False
