"""
https://adventofcode.com/2022/day/19
"""
from itertools import starmap, repeat, islice, product, permutations, combinations
from operator import le, itemgetter, eq
from typing import Iterator, Optional, Iterable
import pytest
import tinyarray as ta
import re
from dataclasses import dataclass
import logging

from more_itertools import first

from utils import dataset_parametrization, DataSetBase


@dataclass
class Blueprint:
    bp_id: int
    costs: tuple[ta.array, ...]


@dataclass
class Inventory:
    resources: ta.array = ta.array((0, 0, 0, 0))
    robots: ta.array = ta.array((1, 0, 0, 0))

    def advance(self, target_robots: tuple[int], blueprint: Blueprint):
        result = Inventory(resources=self.resources, robots=self.robots)
        idx_it = iter(target_robots)
        idx = next(idx_it, 3)
        for minute in range(1, 24 + 1):
            inc = result.robots
            if min(result.resources - blueprint.costs[idx]) >= 0:
                result.resources -= blueprint.costs[idx]
                result.robots += (0,)*idx + (1,) + (0,)*(3-idx)
                idx = next(idx_it, 3)
            result.resources += inc
            # logging.debug("Minute: %d, Inventory: %s", minute, self)
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


round_1 = dataset_parametrization(day="19", examples=[("", 33)], result=None, dataset_class=DataSet)


def generate_target_robots(max_len=24, max_irregular=2):
    for part in combinations(range(max_len), 3):
        result = (0,) * part[0] + (1,) * (part[1]-part[0]) + (2,) * (part[2]-part[1])
        yield result
        if (idx1 := result.index(1)) > 0:
            part1 = permutations(result[:idx1+1])
            rest = result[idx1+1:]
        else:
            part1 = ()
            rest = result
        if (idx2 := rest.index(2)) > 0:
            part2 = permutations(rest[:idx2+1])
            rest = rest[idx2+1:]
        else:
            part2 = ()
        for p1, p2 in product(set(part1), set(part2)):
            result = p1 + p2 + rest
            if result.index(1) < result.inex(2):
                yield result


def quality_level(blueprint: Blueprint) -> int:
    inv = Inventory()
    tlen = 13
    # target_robots = [t + (3,) for t in product((0, 1, 2, 3), repeat=tlen) if
    #                  (1 in t) and (2 in t) and (t.index(1) < t.index(2) < (t.index(3) if 3 in t else tlen))]
    target_robots = list(generate_target_robots(max_len=24))
    stack = [inv.advance(tr, blueprint) for tr in target_robots]
    max_g = max(stack, key=lambda x: x[0].resources[-1])
    logging.info("bp_id: %d, max: %s, target_robots: %s", blueprint.bp_id, max_g[0], max_g[1])
    return blueprint.bp_id * max_g[0].resources[-1]


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    assert sum(map(quality_level, dataset.blueprints())) == dataset.result
