"""
--- Day 9: Smoke Basin ---
https://adventofcode.com/2021/day/9
"""
import numpy as np
import pytest

from adventofcode.event_2021.utils import shift
from adventofcode.utils import dataset_parametrization, DataSetBase

height_map_max = 9


def get_data(input_file: str) -> np.array:
    return np.genfromtxt(input_file, dtype=int, delimiter=1)


def get_low_points(hightmap):
    shifted = np.stack([shift(hightmap, amount=amount, axis=axis) for amount in (1, -1) for axis in (0, 1)])
    return np.logical_and.reduce(shifted - hightmap > 0)


def grow_basin(basin, hightmap):
    shifted = np.stack([shift(basin, amount=amount, axis=axis, fill=False) for amount in (1, -1) for axis in (0, 1)])
    return np.logical_and(np.logical_or(basin, np.logical_or.reduce(shifted)), hightmap != height_map_max)


def max_basin(basin_start, hightmap):
    current_basin = basin_start
    while True:
        larger_basin = grow_basin(current_basin, hightmap)
        if np.array_equal(larger_basin, current_basin):
            return current_basin
        current_basin = larger_basin


def get_basin_size(hightmap, low_point):
    basin_start = np.zeros_like(hightmap, dtype=bool)
    basin_start[low_point] = True
    return max_basin(basin_start, hightmap).sum()


round_1 = dataset_parametrization("2021", "09", [("", 15)], result=522)
round_2 = dataset_parametrization("2021", "09", [("", 1134)], result=916688)


@pytest.mark.parametrize(**round_1)
def test_part_one(dataset: DataSetBase):
    data = dataset.np_array_digits() + 1
    assert data[get_low_points(data)].sum() == dataset.result


@pytest.mark.parametrize(**round_2)
def test_part_two(dataset: DataSetBase):
    data = dataset.np_array_digits()
    low_points = get_low_points(data)
    basin_sizes = [get_basin_size(data, low_point) for low_point in zip(*np.nonzero(low_points))]
    assert np.prod(sorted(basin_sizes)[-3:]) == dataset.result
