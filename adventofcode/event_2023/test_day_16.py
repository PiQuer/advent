"""
--- Day 16: The Floor Will Be Lava ---
https://adventofcode.com/2023/day/16
"""
from collections import deque
from dataclasses import dataclass
from itertools import chain
from multiprocessing import Pool

import numpy as np
import pytest
import tinyarray as ta
from more_itertools import one

from adventofcode.utils import dataset_parametrization, DataSetBase, inbounds

YEAR="2023"
DAY= "16"

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 46)], result=7728, dataset_class=DataSetBase)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 51)], result=8061, dataset_class=DataSetBase)

CACHE = {}
CONTRAPTION = np.array(())

def init(cache_, contraption_):
    # pylint: disable=global-statement
    global CACHE, CONTRAPTION
    CACHE = cache_
    CONTRAPTION = contraption_

@dataclass(frozen=True)
class Beam:
    coordinates: ta.ndarray_int
    direction: ta.ndarray_int

@dataclass
class PropagationResult:
    energized: set[ta.ndarray_int]
    starting_points: set[Beam]

class BeamDeflection(dict):
    def __missing__(self, key):
        return {key}

DEFLECTIONS = {
    b".": BeamDeflection(),
    b"-": BeamDeflection({ta.array((1, 0)): {ta.array((0, -1)), ta.array((0, 1))},
                          ta.array((-1, 0)): {ta.array((0, -1)), ta.array((0, 1))}}),
    b"|": BeamDeflection({ta.array((0, 1)): {ta.array((1, 0)), ta.array((-1, 0))},
                          ta.array((0, -1)): {ta.array((1, 0)), ta.array((-1, 0))}}),
    b"/": BeamDeflection({ta.array((0, 1)): {ta.array((-1, 0))},
                          ta.array((0, -1)): {ta.array((1, 0))},
                          ta.array((1, 0)): {ta.array((0, -1))},
                          ta.array((-1, 0)): {ta.array((0, 1))}}),
    b"\\": BeamDeflection({ta.array((0, 1)): {ta.array((1, 0))},
                           ta.array((0, -1)): {ta.array((-1, 0))},
                           ta.array((1, 0)): {ta.array((0, 1))},
                           ta.array((-1, 0)): {ta.array((0, -1))}})
}

def linear_propagation(start: Beam) -> PropagationResult:
    if start in CACHE:
        return CACHE[start]
    beams = set()
    current = start
    next_beams = set()
    while inbounds(CONTRAPTION.shape, current.coordinates) and current not in beams:
        next_directions = DEFLECTIONS[CONTRAPTION[tuple(current.coordinates)]][current.direction]
        if len(next_directions) == 2:
            next_beams.update(Beam(current.coordinates, d) for d in next_directions)
            break
        beams.add(current)
        current = Beam(current.coordinates + one(next_directions), one(next_directions))
    result = PropagationResult({c.coordinates for c in beams}, next_beams)
    CACHE[start] = result
    return result


def propagate(start: Beam) -> int:
    visited_splitters: set[Beam] = set()
    starting_points: deque[Beam] = deque((start,))
    energized_tiles: set[ta.ndarray_int] = set()
    while starting_points:
        starting_point = starting_points.pop()
        visited_splitters.add(starting_point)
        propagation_result = linear_propagation(starting_point)
        energized_tiles.update(propagation_result.energized)
        starting_points.extend(s for s in propagation_result.starting_points if not s in visited_splitters)
    return len(energized_tiles)

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    init({}, dataset.np_array_bytes())
    assert propagate(Beam(ta.array((0, 0)), ta.array((0, 1)))) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    contraption_ = dataset.np_array_bytes()
    cache_ = {}
    starting_positions = \
        ((Beam(ta.array((p, 0)), ta.array((0, 1))) for p in range(contraption_.shape[0])),
         (Beam(ta.array((p, contraption_.shape[1]-1)), ta.array((0, -1))) for p in range(contraption_.shape[0])),
         (Beam(ta.array((0, p)), ta.array((1, 0))) for p in range(contraption_.shape[1])),
         (Beam(ta.array((contraption_.shape[0]-1, p)), ta.array((-1, 0))) for p in range(contraption_.shape[1])))
    if dataset.id.startswith("example"):
        init(cache_, contraption_)
        assert max(map(propagate, chain.from_iterable(starting_positions))) == dataset.result
    else:
        with Pool(processes=None, initializer=init, initargs=(cache_, contraption_)) as pool:
            assert max(pool.map(propagate, chain.from_iterable(starting_positions))) == dataset.result
