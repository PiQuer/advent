"""
--- Day 18: Boiling Boulders ---
https://adventofcode.com/2022/day/18
"""
from itertools import starmap, combinations
from operator import mul, sub

import numpy as np
import numpy_indexed as npi
import pytest
import tinyarray as ta
from more_itertools import circular_shifts

from utils import dataset_parametrization, DataSetBase, ta_adjacent_3d

round_1 = dataset_parametrization(year="2022", day="18", examples=[("", 64)], result=3662)
round_2 = dataset_parametrization(year="2022", day="18", examples=[("", 58)], result=2060)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    coordinates = np.loadtxt(dataset.input_file, dtype=int, delimiter=",")
    overlapping = get_overlapping_surface(coordinates)
    assert len(coordinates) * 6 - overlapping == dataset.result


def get_overlapping_surface(coordinates):
    overlapping = 0
    for c, c1, c2 in circular_shifts((0, 1, 2)):
        g = npi.group_by(coordinates[:, (c1, c2)])
        overlapping += sum(
            map(np.count_nonzero,
                map(lambda x: x == 1,
                    map(np.diff,
                        map(np.sort, g.split_array_as_list(coordinates[:, c]))))))
    return overlapping * 2


def fill(bounds, coordinates: set[ta.array]) -> set[ta.array]:
    def inbounds(p: ta.array) -> bool:
        return all(b[0] <= c < b[1] for b, c in zip(bounds, p))
    stack = {ta.array(tuple(b[0] for b in bounds))}
    seen = set()
    while stack:
        seen.add(current := stack.pop())
        stack.update(neigh for neigh in (current + adj for adj in ta_adjacent_3d()) if inbounds(neigh)
                     and neigh not in seen and neigh not in coordinates)
    return seen


def surface_of_connected_component(c: set[ta.array]):
    coordinates = np.array(list(c))
    bounds = tuple(zip(np.min(coordinates, axis=0) - 1, np.max(coordinates, axis=0) + 2))
    filled = fill(bounds, c)
    overlapping = get_overlapping_surface(np.array(list(filled)))
    return len(filled) * 6 - overlapping - sum(starmap(mul, combinations(starmap(sub, bounds), 2))) * 2


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    coordinates = set(ta.array(np.loadtxt(dataset.input_file, dtype=int, delimiter=",")))
    assert surface_of_connected_component(coordinates) == dataset.result
