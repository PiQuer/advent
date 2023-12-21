"""
--- Day 17: Clumsy Crucible ---
https://adventofcode.com/2023/day/17
"""
import heapq
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest
import tinyarray as ta
from more_itertools import only, one
from ratelimitingfilter import RateLimitingFilter

from adventofcode.utils import adjacent, inbounds, dataset_parametrization, DataSetBase

ratelimit = RateLimitingFilter(rate=1, per=2, match=["Candidate"])
logging.root.addFilter(ratelimit)


# from utils import generate_rounds

YEAR= "2023"
DAY= "17"

class DataSet(DataSetBase):
    pass

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 102)], result=870, dataset_class=DataSet)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", None)], result=None, dataset_class=DataSet)
# pytest_generate_tests = generate_rounds(round_1, round_2)


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
                   candidates: list[CrucibleWaypoint], target_coordinates: PathCoordinates):
    candidate = heapq.heappop(candidates)
    logging.debug("Candidates: %s, current: %s", len(candidates), candidate)
    neighbors = (PathCoordinates(next_coordinates, ta.array(h))
                 for h in adjacent() if ta.array(h) != -candidate.this.heading and
                 inbounds(data.shape, next_coordinates := candidate.this.coordinates + h))
    for neighbor in neighbors:
        next_straight_segments = 1 if candidate.this.heading != neighbor.heading else \
            candidate.state.straight_segments + 1
        if next_straight_segments > 3:
            continue
        if neighbor.coordinates == target_coordinates.coordinates:
            next_straight_segments = 0
            neighbor = PathCoordinates(neighbor.coordinates, ta.array((0, 0)))
        next_waypoint = \
            CrucibleWaypoint(previous=candidate,
                             state=PathState(value=candidate.state.value + data[*neighbor.coordinates],
                                             straight_segments=next_straight_segments),
                             this=neighbor)
        to_remove = set()
        competing = best_paths[neighbor if neighbor.coordinates != target_coordinates.coordinates else target_coordinates]
        for c in competing:
            if next_waypoint.state < c.state:
                to_remove.add(c)
            elif c.state < next_waypoint.state or c.state == next_waypoint.state:
                break
        else:
            competing -= to_remove
            competing.add(next_waypoint)
            heapq.heappush(candidates, next_waypoint)

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    data = dataset.np_array_digits()
    initial_coordinates = PathCoordinates(heading=ta.array((0, 0)), coordinates=ta.array((0, 0)))
    target_coordinates = PathCoordinates(heading=ta.array((0, 0)), coordinates=ta.array(data.shape) - (1, 1))
    initial_state = PathState(value=0, straight_segments=0)
    best_paths: defaultdict[PathCoordinates, set[CrucibleWaypoint]] = defaultdict(set)
    initial_waypoint = CrucibleWaypoint(previous=None, state=initial_state, this=initial_coordinates)
    best_paths[initial_coordinates] = {initial_waypoint}
    candidates = [initial_waypoint]
    while candidates:
        get_best_paths(best_paths, data, candidates, target_coordinates)
    result = one(best_paths[target_coordinates])
    logging.info("Result: %s", result)
    assert result.state.value == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert dataset.result is None
