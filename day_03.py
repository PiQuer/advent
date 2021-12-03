import numpy as np


def bin_to_int(data: np.array):
    return data.dot(2**np.arange(data.size)[::-1])


def get_data(inputfile):
    data = np.genfromtxt(inputfile, delimiter=1, dtype=np.int8)
    return data


def power_consumption(inputfile: str):
    data = get_data(inputfile)
    gamma = np.round(np.sum(data, axis=0) / data.shape[0]).astype(bool)
    epsilon = np.invert(gamma)
    print(f"power consumption: {bin_to_int(epsilon) * bin_to_int(gamma)}")


def rating(inputfile: str, invert=False) -> int:
    data = get_data(inputfile)
    column = 0
    while data.shape[0] > 1 and column < data.shape[1]:
        mask = data[:, column] == (data[:, column].sum() >= data.shape[0]/2)
        if invert:
            mask = np.invert(mask)
        data = data[mask, :]
        column += 1
    assert data.shape[0] == 1, "This puzzle has no unique solution."
    return bin_to_int(data.squeeze(axis=0))
    pass


def ratings(inputfile: str):
    oxygen_generator_rating = rating(inputfile)
    co2_scrubber_rating = rating(inputfile, invert=True)
    life_support_rating = oxygen_generator_rating * co2_scrubber_rating
    print(f"{oxygen_generator_rating=}, {co2_scrubber_rating=}, {life_support_rating=}")


if __name__ == '__main__':
    power_consumption("input/day_03_example.txt")
    power_consumption("input/day_03.txt")
    ratings("input/day_03_example.txt")
    ratings("input/day_03.txt")
