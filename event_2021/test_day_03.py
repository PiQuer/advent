import numpy as np
import pytest


def bin_to_int(data: np.array):
    return data.dot(2**np.arange(data.size)[::-1])


def get_data(inputfile):
    data = np.genfromtxt(inputfile, delimiter=1, dtype=np.int8)
    return data


def rating(inputfile: str, invert=False) -> int:
    data = get_data(inputfile)
    column = 0
    while data.shape[0] > 1 and column < data.shape[1]:
        mask = data[:, column] == (data[:, column].sum() >= data.shape[0]/2)
        if invert:
            mask = np.invert(mask)
        data = data[mask]
        column += 1
    assert data.shape[0] == 1, "This puzzle has no unique solution."
    return bin_to_int(data.squeeze(axis=0))


@pytest.mark.parametrize("input_file", ["input/day_03_example.txt", "input/day_03.txt"])
class TestDay03:
    def test_power_consumption(self, input_file: str):
        data = get_data(input_file)
        gamma = np.round(np.sum(data, axis=0) / data.shape[0]).astype(bool)
        epsilon = ~gamma
        result = bin_to_int(epsilon) * bin_to_int(gamma)
        print(f"power consumption: {result}")
        assert result == (198 if "example" in input_file else 4191876)

    def test_ratings(self, input_file: str):
        oxygen_generator_rating = rating(input_file)
        co2_scrubber_rating = rating(input_file, invert=True)
        result = oxygen_generator_rating * co2_scrubber_rating
        print(f"{oxygen_generator_rating=}, {co2_scrubber_rating=}, {result=}")
        assert result == (230 if "example" in input_file else 3414905)
