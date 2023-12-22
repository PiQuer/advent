"""
--- Day 21: Step Counter ---
https://adventofcode.com/2023/day/21
"""

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


def next_step(step: int, tiles: set[ta.ndarray_int], seen: dict[ta.ndarray_int, int], data: np.ndarray) -> \
        set[ta.ndarray_int]:
    next_tiles = set()
    for tile in tiles:
        for n in adjacent():
            next_coordinates = tile + n
            shifted_coordinates = (next_coordinates + ((data.shape[0]-1)//2,)*2) % data.shape[0]
            if inbounds(data.shape, shifted_coordinates) and next_coordinates not in seen and \
                    data[*shifted_coordinates] != b'#':
                seen[next_coordinates] = step % 2
                next_tiles.add(next_coordinates)
    return next_tiles

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    data = dataset.np_array_bytes()
    start_position = ta.array((0, 0))
    tiles = {start_position}
    seen = {start_position: 0}
    for step in range(1, dataset.result[0] + 1):
        tiles = next_step(step, tiles, seen, data)
    assert quantify(v == dataset.result[0] % 2 for v in seen.values()) == dataset.result[1]


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    assert dataset.result is None
