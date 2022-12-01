import numpy as np
import pytest

from event_2021.utils import shift

hightmap_max = 9


def get_data(input_file: str) -> np.array:
    return np.genfromtxt(input_file, dtype=int, delimiter=1)


def get_low_points(hightmap):
    shifted = np.stack([shift(hightmap, amount=amount, axis=axis) for amount in (1, -1) for axis in (0, 1)])
    return np.logical_and.reduce(shifted - hightmap > 0)


def grow_basin(basin, hightmap):
    shifted = np.stack([shift(basin, amount=amount, axis=axis, fill=False) for amount in (1, -1) for axis in (0, 1)])
    return np.logical_and(np.logical_or(basin, np.logical_or.reduce(shifted)), hightmap != hightmap_max)


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


@pytest.mark.parametrize("input_file", ["input/day_09_example.txt", "input/day_09.txt"])
class TestDay09:
    def test_part_one(self, input_file: str):
        data = get_data(input_file) + 1
        result = data[get_low_points(data)].sum()
        print(f"The sum of the risk levels is {result}.")
        assert result == (15 if "example" in input_file else 522)

    def test_part_two(self, input_file: str):
        data = get_data(input_file)
        low_points = get_low_points(data)
        basin_sizes = [get_basin_size(data, low_point) for low_point in zip(*np.nonzero(low_points))]
        result = np.prod(sorted(basin_sizes)[-3:])
        print(f"The product of the three largest basin sizes is {result}")
        assert result == (1134 if "example" in input_file else 916688)
