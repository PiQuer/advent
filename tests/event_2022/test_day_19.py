"""
--- Day 19: Not Enough Minerals ---
https://adventofcode.com/2022/day/19
"""
import heapq
import logging
import re
from dataclasses import dataclass
from itertools import islice
from typing import Iterator, Optional

import pytest
import tinyarray as ta
from ratelimitingfilter import RateLimitingFilter

from utils import dataset_parametrization, DataSetBase

ratelimit = RateLimitingFilter(rate=1, per=2, match=["Candidate"])
logging.root.addFilter(ratelimit)


@dataclass
class Blueprint:
    bp_id: int
    costs: tuple[ta.array, ...]


@dataclass(frozen=True)
class State:
    resources: ta.array
    buy: tuple[int]
    robots: ta.array
    not_increased_since: int = 0
    total_decreased: int = 0
    previous: Optional["State"] = None

    def __hash__(self):
        return hash(tuple(self.buy.count(x) for x in range(4)))

    def __eq__(self, other: "State"):
        return tuple(self.buy.count(x) for x in range(4)) == tuple(other.buy.count(x) for x in range(4))

    def __lt__(self, other: "State"):
        return self.resources > other.resources

    def __str__(self):
        return f"resources={tuple(self.resources)}, buy={self.buy}, robots={tuple(self.robots)}, " \
               f"ni={self.not_increased_since}, previous_buy={self.previous.buy if self.previous else 'None'}, " \
               f"resources_diff={self.resources - self.previous.resources if self.previous else 'None'}"


def evaluate(buy: tuple[int, ...], blueprint: Blueprint, minutes, previous: Optional[State] = None) -> State:
    resources = ta.array((0, 0, 0, 0))
    robots = ta.array((0, 0, 0, 1))
    idx_it = iter(buy)
    idx = next(idx_it, 3)
    for minute in range(1, minutes + 1):
        inc = robots
        if min(resources - blueprint.costs[idx]) >= 0:
            resources -= blueprint.costs[idx]
            robots += (0,)*(3-idx) + (1,) + (0,)*idx
            idx = next(idx_it, 3)
        resources += inc
    if previous is None:
        total_decreased = not_increased = 0
    else:
        total_decreased = previous.total_decreased + (1 if tuple(resources)[:3] < tuple(previous.resources)[:3] else 0)
        not_increased = (previous.not_increased_since + 1) if tuple(resources)[:3] <= tuple(previous.resources)[:3] \
            else 0
    return State(resources=resources, buy=buy, robots=robots, not_increased_since=not_increased, previous=previous,
                 total_decreased=total_decreased)


def generate_next(buy: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    seen = set()
    first_one = buy.index(1)
    first_two = buy.index(2)
    first_three = buy.index(3) if 3 in buy else len(buy)
    for n in range(4):
        r = range(0)
        if n == 0:
            r = range(first_two)
        elif n == 1:
            r = range(first_three)
        elif n == 2:
            r = range(first_one + 1, len(buy) + 1)
        elif n == 3:
            r = range(max(first_two + 1, first_one + 2), len(buy))
        for pos in r:
            if 0 < pos < len(buy) and buy[pos-1] == n == buy[pos]:
                continue
            result = buy[:pos] + (n,) + buy[pos:]
            if result not in seen:
                seen.add(result)
                if result[-1] in (0, 3) or \
                        (result.index(2) < result.index(1)) or \
                        (3 in result and result.index(3) < result.index(2)):
                    continue
                yield result


class DataSet(DataSetBase):
    def blueprints(self) -> Iterator[Blueprint]:
        regex = r"Blueprint (\d+):.*costs (\d+) ore.*costs (\d+) ore.*costs (\d+) ore and (\d+) clay.*" \
                r"costs (\d+) ore and (\d+) obsidian"
        for line in self.lines():
            match = re.search(regex, line)
            yield Blueprint(bp_id=int(match.group(1)),
                            costs=(ta.array((0, 0, 0, int(match.group(2)))),
                                   ta.array((0, 0, 0, int(match.group(3)))),
                                   ta.array((0, 0, int(match.group(5)), int(match.group(4)))),
                                   ta.array((0, int(match.group(7)), 0, int(match.group(6))))))


round_1 = dataset_parametrization(year="2022", day="19", examples=[("", 33)], result=790, dataset_class=DataSet)
round_2 = dataset_parametrization(year="2022", day="19", examples=[("", 56 * 62)], result=7350, dataset_class=DataSet)


def get_next_states(candidates: list[State], seen: dict[State, tuple[int, ...]], maximum: State,
                    blueprint: Blueprint, minutes: int, counter: int) -> tuple[State, int]:
    c = heapq.heappop(candidates)
    counter += 1
    logging.debug("Candidate: %s", c)
    for t in generate_next(c.buy):
        next_candidate = evaluate(t, blueprint, minutes, c)
        if next_candidate.total_decreased >= 2:
            continue
        elif next_candidate.not_increased_since >= 3:
            continue
        if next_candidate in seen and seen[next_candidate] >= next_candidate.resources:
            continue
        seen[next_candidate] = next_candidate.resources
        if next_candidate.resources > maximum.resources:
            maximum = next_candidate
            logging.debug("New max at counter %d: %s", counter, maximum)
        if len(next_candidate.buy) <= minutes:
            heapq.heappush(candidates, next_candidate)
    return maximum, counter


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    minutes = 24
    quality_level = 0
    for blueprint in dataset.blueprints():
        candidates = list(evaluate(x, blueprint, minutes) for x in generate_next((1, 2)))
        heapq.heapify(candidates)
        seen = {initial: initial.resources for initial in candidates}
        maximum = candidates[0]
        counter = 0
        while candidates and counter < 2000:
            maximum, counter = get_next_states(candidates, seen, maximum, blueprint, minutes, counter)
        quality_level += blueprint.bp_id * maximum.resources[0]
        logging.info("Blueprint id: %d, maximum state: %s", blueprint.bp_id, maximum)
    assert quality_level == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    minutes = 32
    result = 1
    for blueprint in islice(dataset.blueprints(), 3):
        evaluate((0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 3, 2, 2, 3, 2, 3), blueprint, minutes)
        candidates = list(evaluate(x, blueprint, minutes) for x in generate_next((1, 2)))
        heapq.heapify(candidates)
        seen = {initial: initial.resources for initial in candidates}
        maximum = candidates[0]
        counter = 0
        while candidates and counter < 2000:
            maximum, counter = get_next_states(candidates, seen, maximum, blueprint, minutes, counter)
        result *= maximum.resources[0]
        logging.info("Blueprint id: %d, maximum state: %s", blueprint.bp_id, maximum)
    assert result == dataset.result
