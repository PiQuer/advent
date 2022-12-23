"""
https://adventofcode.com/2022/day/19
TODO: not a general solution, and too slow
"""
from functools import partial, reduce
from itertools import starmap, combinations, pairwise, chain, islice
from operator import le, mul
from typing import Iterator
import pytest
import tinyarray as ta
import re
from dataclasses import dataclass
import logging

from utils import dataset_parametrization, DataSetBase


@dataclass
class Blueprint:
    bp_id: int
    costs: tuple[ta.array, ...]


@dataclass
class Inventory:
    resources: ta.array = ta.array((0, 0, 0, 0))
    robots: ta.array = ta.array((1, 0, 0, 0))

    def advance(self, target_robots: tuple[int], blueprint: Blueprint, minutes=24):
        result = Inventory(resources=self.resources, robots=self.robots)
        idx_it = iter(target_robots)
        idx = next(idx_it, 3)
        for minute in range(1, minutes + 1):
            inc = result.robots
            if min(result.resources - blueprint.costs[idx]) >= 0:
                result.resources -= blueprint.costs[idx]
                result.robots += (0,)*idx + (1,) + (0,)*(3-idx)
                idx = next(idx_it, 3)
            result.resources += inc
        return result, target_robots

    def __le__(self, other: "Inventory"):
        result = all(starmap(le, zip(self.resources, other.resources))) \
            and all(starmap(le, zip(self.robots, other.robots)))
        if result:
            logging.debug("%s <= %s", self, other)
        return result

    def __hash__(self):
        return hash((self.resources, self.robots))


class DataSet(DataSetBase):
    def blueprints(self) -> Iterator[Blueprint]:
        regex = r"Blueprint (\d+):.*costs (\d+) ore.*costs (\d+) ore.*costs (\d+) ore and (\d+) clay.*" \
                r"costs (\d+) ore and (\d+) obsidian"
        for line in self.lines():
            match = re.search(regex, line)
            yield Blueprint(bp_id=int(match.group(1)),
                            costs=(ta.array((int(match.group(2)), 0, 0, 0)),
                                   ta.array((int(match.group(3)), 0, 0, 0)),
                                   ta.array((int(match.group(4)), int(match.group(5)), 0, 0)),
                                   ta.array((int(match.group(6)), 0, int(match.group(7)), 0))))


round_1 = dataset_parametrization(day="19", examples=[("", 33)], result=790, dataset_class=DataSet)
round_2 = dataset_parametrization(day="19", examples=[("", 56 * 62)], result=7350, dataset_class=DataSet)


def irregularity(target_robots: tuple[int, ...], irregularities: int, max_len: int):
    if irregularities == 0:
        yield target_robots
    ltr = len(target_robots)
    if ltr + irregularities <= max_len:
        for idx in combinations(range(ltr - 1), irregularities):
            if idx:
                yield target_robots[:idx[0]] + \
                      sum(((target_robots[a] + 1,) + target_robots[a:b] for a, b in pairwise(idx + (ltr,))), ())


def generate_target_robots(max_len=24, max_irregular=2):
    for part in combinations(range(max_len), 3):
        result = (0,) * part[0] + (1,) * (part[1]-part[0]) + (2,) * (part[2]-part[1]) + (3,)
        yield from \
            (g for g in chain.from_iterable(irregularity(result, ni, max_len) for ni in range(max_irregular + 1))
             if g.index(1) < g.index(2) < g.index(3))


def max_geodes(blueprint: Blueprint, minutes: int = 24, tr_len: int = 20, ir: int = 3) -> int:
    inv = Inventory()
    target_robots = generate_target_robots(tr_len, ir)
    stack = [inv.advance(tr, blueprint, minutes) for tr in target_robots]
    max_g = max(stack, key=lambda x: x[0].resources[-1])
    logging.info("bp_id: %d, max: %s, target_robots: %s", blueprint.bp_id, max_g[0], max_g[1])
    return max_g[0].resources[-1]


def quality_level(blueprint: Blueprint, minutes: int = 24, tr_len: int = 20, ir: int = 3) -> int:
    return max_geodes(blueprint, minutes, tr_len, ir) * blueprint.bp_id


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    assert sum(map(partial(quality_level, tr_len=16, ir=2), dataset.blueprints())) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert reduce(mul, map(partial(max_geodes, tr_len=23, ir=4, minutes=32), islice(dataset.blueprints(), 3))) \
        == dataset.result
