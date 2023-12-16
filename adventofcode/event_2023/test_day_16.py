"""
--- Day 16: The Floor Will Be Lava ---
https://adventofcode.com/2023/day/16
"""
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pytest
import tinyarray as ta

from adventofcode.utils import dataset_parametrization, DataSetBase, inbounds

# from utils import generate_rounds

YEAR="2023"
DAY= "16"

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 46)], result=7728, dataset_class=DataSetBase)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", None)], result=None, dataset_class=DataSetBase)
# pytest_generate_tests = generate_rounds(round_1, round_2)


@dataclass(frozen=True)
class Beam:
    coordinates: ta.ndarray_int
    direction: ta.ndarray_int


PROPAGATION = {
    b".": {},
    b"-": {ta.array((1, 0)): {ta.array((0, -1)), ta.array((0, 1))},
          ta.array((-1, 0)): {ta.array((0, -1)), ta.array((0, 1))}},
    b"|": {ta.array((0, 1)): {ta.array((1, 0)), ta.array((-1, 0))},
          ta.array((0, -1)): {ta.array((1, 0)), ta.array((-1, 0))}},
    b"/": {ta.array((0, 1)): {ta.array((-1, 0))},
          ta.array((0, -1)): {ta.array((1, 0))},
          ta.array((1, 0)): {ta.array((0, -1))},
          ta.array((-1, 0)): {ta.array((0, 1))}},
    b"\\": {ta.array((0, 1)): {ta.array((1, 0))},
            ta.array((0, -1)): {ta.array((-1, 0))},
            ta.array((1, 0)): {ta.array((0, 1))},
            ta.array((-1, 0)): {ta.array((0, -1))}}
}

def propagate(start: Beam, contraption: np.ndarray) -> defaultdict[ta.ndarray_int, set[Beam]]:
    result = defaultdict(set)
    result[ta.array((0, 0))].add(start)
    beams = deque((start,))
    while beams:
        beam = beams.pop()
        for next_step in PROPAGATION[contraption[*beam.coordinates]].get(beam.direction, {beam.direction}):
            if inbounds(contraption.shape, next_coordinates := beam.coordinates + next_step):
                if (next_beam := Beam(next_coordinates, next_step)) not in result[next_coordinates]:
                    beams.append(next_beam)
                    result[next_coordinates].add(next_beam)
    return result

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    assert len(propagate(Beam(ta.array((0, 0)), ta.array((0, 1))), dataset.np_array_bytes())) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    pass
