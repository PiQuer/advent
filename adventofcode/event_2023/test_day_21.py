"""
--- Day 21: Step Counter ---
https://adventofcode.com/2023/day/21
"""
import math
from dataclasses import dataclass
from functools import cache

import numpy as np
import pytest
import tinyarray as ta
from more_itertools import quantify

from adventofcode.utils import dataset_parametrization, DataSetBase, adjacent, inbounds

YEAR= "2023"
DAY= "21"

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", (6, 16))], result=(64, 3617))
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", (6, 16)), ("", (10, 50)), ("", (50, 1594)),
                                                                ("", (100, 6536)), ("", (500, 167004)),
                                                                ("", (1000, 668697)), ("", (5000, 16733044))],
                                  result=(26501365, None))

@dataclass
class Coordinates:
    coordinates: ta.ndarray_int
    data: np.ndarray

    def __hash__(self):
        return hash((self.coordinates, self.data.shape))

    def __eq__(self, other: "Coordinates") -> bool:
        return self.coordinates == other.coordinates and self.data.shape == other.data.shape


@cache
def next_coordinates_good(next_coordinates: Coordinates) -> ta.ndarray_int | None:
    return inbounds(next_coordinates.data.shape, next_coordinates.coordinates) and \
            next_coordinates.data[*next_coordinates.coordinates] != b'#'


def next_step(step: int, tiles: set[Coordinates], seen: dict[Coordinates, int]) -> \
        set[ta.ndarray_int]:
    next_tiles = set()
    for tile in tiles:
        for n in adjacent():
            next_coordinates = Coordinates(tile.coordinates + n, tile.data)
            if next_coordinates_good(next_coordinates) and step < seen.get(next_coordinates, math.inf):
                seen[next_coordinates] = step
                next_tiles.add(next_coordinates)
    return next_tiles

@cache
def fill_garden(starting_position: Coordinates) -> dict[Coordinates, int]:
    tiles = {starting_position}
    seen = {starting_position: 0}
    step = 0
    while tiles:
        step += 1
        tiles = next_step(step, tiles, seen)
    return seen

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    data = dataset.np_array_bytes
    garden = fill_garden(Coordinates((ta.array(data.shape) - (1, 1)) // 2, data))
    steps = dataset.result[0]
    assert quantify(v <= steps and v % 2 == steps % 2 for v in garden.values()) == dataset.result[1]


@dataclass
class Garden:
    base_coordinates: ta.ndarray_int
    offset_north = ta.ndarray_int
    offset_west = ta.ndarray_int
    offset_south = ta.ndarray_int
    offset_east = ta.ndarray_int

@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    garden = fill_garden(Coordinates(ta.array((0, 0)), dataset.np_array_bytes))
