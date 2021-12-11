import numpy as np
import pytest


num_steps = 100
window = (slice(1, -1), slice(1, -1))


def get_data(input_file: str):
    data = np.genfromtxt(input_file, dtype=int, delimiter=1)
    result = np.zeros(np.array(data.shape) + 2, dtype=int)
    result[1:-1, 1:-1] = data
    return result


def shift(array: np.array, axes):
    slices = {-1: (2, None), 0: (1, -1), 1: (None, -2)}
    return array[tuple([slice(*slices[a]) for a in axes])]


def flash(data):
    adjacents = np.stack([shift(data > 9, (x, y)) for x in range(-1, 2) for y in range(-1, 2)]).sum(axis=0)
    mask = np.logical_or(data[window] > 9, data[window] == 0)
    data[window] += adjacents * (~mask)
    data[window][mask] = 0
    return data


def calculate_step(data):
    data[window] += 1
    while True:
        new = flash(np.copy(data))
        if np.array_equal(new, data):
            return
        data[...] = new


@pytest.mark.parametrize("input_file", ("input/day_11_example.txt", "input/day_11.txt"))
def test_part_one(input_file):
    data = get_data(input_file)
    result = 0
    for _ in range(num_steps):
        calculate_step(data)
        result += (data[window] == 0).sum()
    assert result == (1656 if "example" in input_file else 1665)


@pytest.mark.parametrize("input_file", ("input/day_11_example.txt", "input/day_11.txt"))
def test_part_two(input_file):
    data = get_data(input_file)
    counter = 0
    while data[window].sum() > 0:
        calculate_step(data)
        counter += 1
    assert counter == (195 if "example" in input_file else 235)
