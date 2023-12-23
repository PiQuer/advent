"""
--- Day 22: Sand Slabs ---
https://adventofcode.com/2023/day/22
"""
import heapq
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from operator import add

import portion as P
import pytest
from more_itertools import quantify, first

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
        return self.z.upper < other.z.upper

class DataSet(DataSetBase):
    def line_to_brick(self, line) -> Brick:
        lower, upper = line.split('~')
        return Brick(*(P.closedopen(*interval)
                     for interval in zip(map(int, lower.split(',')), map(add, map(int, upper.split(',')), repeat(1)))))

    def bricks(self) -> list[Brick]:
        return list(map(self.line_to_brick, self.lines()))

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 5)], result=421, dataset_class=DataSet)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 7)], result=39247, dataset_class=DataSet)


def fall(bricks: list[Brick]):
    bricks.sort()
    for i, brick in enumerate(bricks):
        if brick.z.lower == 1:
            continue
        fallen = brick
        for j in range(i-1, -1, -1):
            other = bricks[j]
            if other.z.upper < fallen.z.lower:
                fallen = fallen.move_to_z(other.z.upper)
            if other.supports(fallen):
                bricks[i] = fallen
                break
        else:
            bricks[i] = fallen.move_to_z(1)
        bricks[j:i+1] = sorted(bricks[j:i+1])

def get_supporters(bricks: list[Brick]) -> tuple[defaultdict[Brick, set[Brick]], defaultdict[Brick, set[Brick]]]:
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


def get_falling(brick: Brick, supports, supported_by) -> int:
    falling = {brick}
    current_layer: set[Brick] = set()
    candidates = [brick]
    while candidates:
        while candidates:
            candidate = heapq.heappop(candidates)
            if not current_layer or candidate.z.upper == first(current_layer).z.upper:
                current_layer.add(candidate)
            else:
                heapq.heappush(candidates, candidate)
                break
        falling.update(current_layer)
        seen = set()
        for b in current_layer:
            for k in supported_by[b]:
                if (k not in seen) and (not supports[k] - falling):
                    seen.add(k)
                    heapq.heappush(candidates, k)
        current_layer = set()
    return len(falling) - 1

@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    bricks = dataset.bricks()
    fall(bricks)
    supports, supported_by = get_supporters(bricks)
    assert sum(map(partial(get_falling, supports=supports, supported_by=supported_by), bricks)) == dataset.result
