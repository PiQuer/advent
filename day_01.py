import numpy as np


def count_increasing(data):
    return (data[1:] - data[:-1] > 0).sum()


def part_one(input_file: str):
    data = np.genfromtxt(input_file, dtype=int)
    print(f"Part one increasing: {count_increasing(data)}")


def part_two(input_file: str):
    data = np.lib.stride_tricks.sliding_window_view(np.genfromtxt(input_file, dtype=int), window_shape=(3, ))
    print(f"Part two increasing: {count_increasing(data.sum(axis=1))}")


if __name__ == '__main__':
    part_one("input/day_01_example.txt")
    part_one("input/day_01.txt")
    part_two("input/day_01_example.txt")
    part_two("input/day_01.txt")
