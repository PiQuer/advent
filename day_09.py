import numpy as np

hightmap_max = 9


def get_data(input_file: str) -> np.array:
    return np.genfromtxt(input_file, dtype=int, delimiter=1)


def shift(array: np.array, amount=1, axis=0, fill=11):
    result = np.roll(np.copy(array), amount, axis=axis)
    index = [slice(None) for _ in result.shape]
    index[axis] = slice(min(np.sign(amount), 0), amount + min(np.sign(amount), 0), np.sign(amount))
    result[tuple(index)] = fill
    return result


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


def part_one(input_file: str):
    data = get_data(input_file) + 1
    print(f"The sum of the risk levels is {data[get_low_points(data)].sum()}.")


def part_two(input_file: str):
    data = get_data(input_file)
    low_points = get_low_points(data)
    basin_sizes = [get_basin_size(data, low_point) for low_point in zip(*np.nonzero(low_points))]
    print(f"The product of the three largest basin sizes is {np.prod(sorted(basin_sizes)[-3:])}")


if __name__ == '__main__':
    [part_one(input_file) for input_file in ("input/day_09_example.txt", "input/day_09.txt")]
    [part_two(input_file) for input_file in ("input/day_09_example.txt", "input/day_09.txt")]
