import numpy as np
import pytest


def count_increasing(data):
    return (data[1:] - data[:-1] > 0).sum()


@pytest.mark.parametrize("input_file", ["input/day_01_example.txt", "input/day_01.txt"])
class TestDay01:
    def test_part_one(self, input_file: str):
        data = np.genfromtxt(input_file, dtype=int)
        result = count_increasing(data)
        print(f"Part one increasing: {result}")
        assert result == (7 if "example" in input_file else 1448)

    def test_part_two(self, input_file: str):
        data = np.lib.stride_tricks.sliding_window_view(np.genfromtxt(input_file, dtype=int), window_shape=(3, ))
        result = count_increasing(data.sum(axis=1))
        print(f"Part two increasing: {result}")
        assert result == (5 if "example" in input_file else 1471)
