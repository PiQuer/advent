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

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 16, {'steps': 6})], steps=64, part=1)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[], steps=26501365, part=2)

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
def fill_garden(starting_position: Coordinates, max_steps: int) -> tuple[int, int]:
    if max_steps < 0:
        return 0
    tiles = {starting_position}
    seen = {starting_position: 0}
    step = 1
    while tiles and step <= max_steps:
        tiles = next_step(step, tiles, seen)
        step += 1
    return quantify(v % 2 == max_steps % 2 for v in seen.values()), \
        quantify(v <= max_steps-1 and v % 2 == (max_steps-1)%2 for v in seen.values())

@pytest.fixture(autouse=True)
def clear_caches():
    fill_garden.cache_clear()
    next_coordinates_good.cache_clear()

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    next_coordinates_good.cache_clear()
    data = dataset.np_array_bytes
    steps = dataset.params['steps']
    result, _ = fill_garden(Coordinates((ta.array(data.shape) - (1, 1)) // 2, data), max_steps=steps)
    dataset.assert_answer(result)


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    data = dataset.np_array_bytes
    start = Coordinates((ta.array(data.shape) - (1, 1)) // 2, data)
    steps = dataset.params['steps']
    p = data.shape[0]
    y = (steps - ((p - 1) // 2)) // p
    count_odd, count_even = fill_garden(start, p-1)
    nw_count_odd, nw_count_even = fill_garden(Coordinates(ta.array((0, 0)), data), max_steps=(p-1)//2-1)
    sw_count_odd, sw_count_even = fill_garden(Coordinates(ta.array((p-1, 0)), data), max_steps=(p-1)//2-1)
    ne_count_odd, ne_count_even = fill_garden(Coordinates(ta.array((0, p-1)), data), max_steps=(p-1)//2-1)
    se_count_odd, se_count_even = fill_garden(Coordinates(ta.array((p-1, p-1)), data), max_steps=(p-1)//2-1)
    if y % 2 == 0:
        full_tiles_even = (y - 1) ** 2
        full_tiles_odd = y ** 2
        count = full_tiles_even * count_even + full_tiles_odd * count_odd \
            + y * (nw_count_odd + sw_count_odd + ne_count_odd + se_count_odd) \
            + (y-1) * (4*count_even - nw_count_even - sw_count_even - ne_count_even - se_count_even) \
            + 4*count_even - 2*(nw_count_even + sw_count_even + ne_count_even + se_count_even)
    else:
        full_tiles_even = y ** 2
        full_tiles_odd = (y-1) ** 2
        count = full_tiles_even * count_even + full_tiles_odd * count_odd \
                + y * (nw_count_even + sw_count_even + ne_count_even + se_count_even) \
                + (y - 1) * (4 * count_odd - nw_count_odd - sw_count_odd - ne_count_odd - se_count_odd) \
                + 4 * count_odd - 2 * (nw_count_odd + sw_count_odd + ne_count_odd + se_count_odd)
    dataset.assert_answer(count)
