"""
--- Day 22: Sand Slabs ---
https://adventofcode.com/2023/day/22
"""
from collections import defaultdict
from dataclasses import dataclass
from itertools import repeat
from operator import add

import portion as P
import pytest
from more_itertools import quantify

from adventofcode.utils import dataset_parametrization, DataSetBase

YEAR= "2023"
DAY= "22"

@dataclass(frozen=True)
class Brick:
    x: P.Interval
    y: P.Interval
    z: P.Interval

    def supports(self, other: "Brick") -> bool:
        return self.z.upper == other.z.lower \
            and bool(self.x.intersection(other.x)) and bool(self.y.intersection(other.y))

    def move_to_z(self, z: int) -> "Brick":
        if z == self.z.lower:
            return self
        return Brick(x=self.x, y=self.y, z=P.closedopen(z, z + (self.z.upper - self.z.lower)))

    def __lt__(self, other: "Brick") -> bool:
        if self.z.upper < other.z.upper:
            return True
        if self.z.upper > other.z.upper:
            return False
        return self.z.lower < other.z.lower

    def sort_by_upper(self):
        return (self.z.upper, self.z.lower)

    def sort_by_lower(self):
        return (self.z.lower, self.z.upper)

class DataSet(DataSetBase):
    def line_to_brick(self, line) -> Brick:
        lower, upper = line.split('~')
        return Brick(*(P.closedopen(*interval)
                     for interval in zip(map(int, lower.split(',')), map(add, map(int, upper.split(',')), repeat(1)))))

    def bricks(self) -> list[Brick]:
        return list(map(self.line_to_brick, self.lines()))

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 5)], result=421, dataset_class=DataSet)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 7)], result=None, dataset_class=DataSet)


def fall(bricks: list[Brick]):
    bricks.sort(key=Brick.sort_by_upper)
    for i, brick in enumerate(bricks):
        if brick.z.lower == 1:
            continue
        fallen = brick
        for other in bricks[i-1::-1]:
            if other.z.upper < fallen.z.lower:
                fallen = fallen.move_to_z(other.z.upper)
            if other.supports(fallen):
                bricks[i] = fallen
                break
        else:
            bricks[i] = fallen.move_to_z(1)
        bricks[:i+1] = sorted(bricks[:i+1], key=Brick.sort_by_upper)

def get_supporters(bricks: list[Brick]) -> tuple[defaultdict[Brick, set[Brick]], defaultdict[Brick, set[Brick]]]:
    bricks.sort(key=Brick.sort_by_upper)
    supports: defaultdict[Brick, set[Brick]] = defaultdict(set)
    supported_by: defaultdict[Brick, set[Brick]] = defaultdict(set)
    for i, brick in enumerate(reversed(bricks)):
        for other in bricks[-i-1::-1]:
            if brick.z.lower > other.z.upper:
                break
            if other.supports(brick):
                supports[brick].add(other)
                supported_by[other].add(brick)
    return supports, supported_by


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    bricks = dataset.bricks()
    fall(bricks)
    supports, supported_by = get_supporters(bricks)
    result = quantify(all(supports[supported] - {brick} for supported in supported_by[brick]) for brick in bricks)
    assert result == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert dataset.result is None
