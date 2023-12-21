"""
--- Day 17: Clumsy Crucible ---
https://adventofcode.com/2023/day/17
"""
import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import tinyarray as ta
from more_itertools import one
from ratelimitingfilter import RateLimitingFilter

from adventofcode.utils import adjacent, inbounds, dataset_parametrization, DataSetBase, generate_rounds

ratelimit = RateLimitingFilter(rate=1, per=2, match=["Candidate"])
logging.root.addFilter(ratelimit)


YEAR= "2023"
DAY= "17"

class DataSet(DataSetBase):
    pass

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 102)], result=870, dataset_class=DataSet,
                                  initial_straight=1, max_straight=3)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 94)], result=1063, dataset_class=DataSet,
                                  initial_straight=4, max_straight=10)
pytest_generate_tests = generate_rounds(round_1, round_2)

@dataclass(frozen=True)
class PathCoordinates:
    coordinates: ta.ndarray_int
    heading: ta.ndarray_int

@dataclass(frozen=True)
class PathState:
    value: int
    straight_segments: int

    def __lt__(self, other: "PathState"):
        return (self.value < other.value and self.straight_segments <= other.straight_segments) \
            or (self.value == other.value) and self.straight_segments < other.straight_segments

    def __eq__(self, other: "PathState"):
        return self.value == other.value and self.straight_segments == other.straight_segments

@dataclass(frozen=True)
class CrucibleWaypoint:
    previous: Optional["CrucibleWaypoint"] = field(hash=False, compare=False)
    state: PathState
    this: PathCoordinates

    def __lt__(self, other: "CrucibleWaypoint"):
        return self.state.value < other.state.value

    def __str__(self):
        previous_coordinates = self.previous.this.coordinates if self.previous is not None else None
        return f"Previous: {previous_coordinates}, This: {self.this.coordinates}, " \
               f"Heading: {self.this.heading}, Straight: {self.state.straight_segments}, Value: {self.state.value})"

    def __repr__(self):
        return self.__str__()

def visualize_path(data: np.ndarray, waypoint: CrucibleWaypoint):
    visualize = data.copy()
    while waypoint is not None:
        visualize[*waypoint.this.coordinates] += 100
        waypoint = waypoint.previous
    return visualize


def get_best_paths(best_paths: defaultdict[PathCoordinates, set[CrucibleWaypoint]], data: np.ndarray,
                   candidates: list[CrucibleWaypoint], target_coordinates: PathCoordinates,
                   initial_straight: int, max_straight: int):
    candidate = heapq.heappop(candidates)
    logging.debug("Candidates: %s, current: %s", len(candidates), candidate)
    for h in (ah for _ in adjacent() if (ah := ta.array(_)) != -candidate.this.heading):
        if candidate.this.heading == h:
            next_coordinates = candidate.this.coordinates + h
            next_straight_segments = candidate.state.straight_segments + 1
            if (not inbounds(data.shape, next_coordinates)) or next_straight_segments > max_straight:
                continue
            next_value = candidate.state.value + data[*(candidate.this.coordinates + h)]
        else:
            next_coordinates = candidate.this.coordinates + initial_straight * h
            next_straight_segments = initial_straight
            if not inbounds(data.shape, next_coordinates):
                continue
            next_value = candidate.state.value + \
                         sum(data[*(candidate.this.coordinates + (r+1) * h)] for r in range(initial_straight))
        if next_coordinates == target_coordinates.coordinates:
            next_straight_segments = 0
            neighbor = PathCoordinates(target_coordinates.coordinates, ta.array((0, 0)))
        else:
            neighbor = PathCoordinates(next_coordinates, h)
        next_waypoint = \
            CrucibleWaypoint(previous=candidate,
                             state=PathState(value=next_value, straight_segments=next_straight_segments),
                             this=neighbor)
        to_remove = set()
        competing = best_paths[neighbor]
        for c in competing:
            if next_waypoint.state < c.state:
                to_remove.add(c)
            elif c.state < next_waypoint.state or c.state == next_waypoint.state:
                break
        else:
            competing -= to_remove
            competing.add(next_waypoint)
            heapq.heappush(candidates, next_waypoint)

def test_day_17(dataset: DataSet):
    data = dataset.np_array_digits()
    initial_coordinates = PathCoordinates(heading=ta.array((0, 0)), coordinates=ta.array((0, 0)))
    target_coordinates = PathCoordinates(heading=ta.array((0, 0)), coordinates=ta.array(data.shape) - (1, 1))
    initial_state = PathState(value=0, straight_segments=0)
    best_paths: defaultdict[PathCoordinates, set[CrucibleWaypoint]] = defaultdict(set)
    initial_waypoint = CrucibleWaypoint(previous=None, state=initial_state, this=initial_coordinates)
    best_paths[initial_coordinates] = {initial_waypoint}
    candidates = [initial_waypoint]
    while candidates:
        get_best_paths(best_paths, data, candidates, target_coordinates, **dataset.params)
    result = one(best_paths[target_coordinates])
    logging.info("Result: %s", result)
    assert result.state.value == dataset.result
