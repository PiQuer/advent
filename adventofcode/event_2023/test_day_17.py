"""
--- Day 17: Clumsy Crucible ---
https://adventofcode.com/2023/day/17
"""
from dataclasses import dataclass

import numpy as np
import pytest
import tinyarray as ta

from adventofcode.utils import dataset_parametrization, DataSetBase
from utils import Waypoint, adjacent, inbounds

# from utils import generate_rounds

YEAR= "2023"
DAY= "17"

class DataSet(DataSetBase):
    pass

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 102)], result=None, dataset_class=DataSet)
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
        return self.value < other.value and self.straight_segments <= other.straight_segments

@dataclass(frozen=True)
class CrucibleWaypoint:
    length: int
    previous: PathCoordinates | None
    state: PathState

def get_best_paths(length: int, best_paths: dict[ta.ndarray_int, set[CrucibleWaypoint]], data: np.ndarray,
                   candidates: set[PathCoordinates]) -> set[PathCoordinates]:
    next_candidates: set[PathCoordinates] = set()
    for candidate in candidates:
        neighbors = (PathCoordinates(next_coordinates, h) for h in adjacent()
                     if inbounds(data.shape, next_coordinates := candidate.coordinates + h))
        for neighbor in neighbors:
            pass
    return next_candidates

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    data = dataset.np_array()
    initial_coordinates = PathCoordinates(heading=ta.ndarray_int((0, 0)), coordinates=ta.ndarray_int((0, 0)))
    initial_state = PathState(value=0, straight_segments=0)
    best_paths = {initial_coordinates: {CrucibleWaypoint(length=0, previous=None, state=initial_state)}}
    length = 0
    candidates = {ta.array((0, 0))}
    while candidates:
        length += 1
        candidates = get_best_paths(length, best_paths, data, candidates)
    assert best_paths[ta.array(data.shape) - (1, 1)].value == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert dataset.result is None
