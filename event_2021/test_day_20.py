import pytest
from pathlib import Path
import numpy as np


def get_data(input_file):
    data = Path(input_file).read_text().split('\n\n')
    enhancement_algorithm = (np.array(list(data[0])) == '#').astype(int)
    image = (np.array([list(line) for line in data[1].splitlines()]) == '#').astype(int)
    return enhancement_algorithm, image


def shift(array: np.array, axes):
    slices = {-1: (2, None), 0: (1, -1), 1: (None, -2)}
    return array[tuple([slice(*slices[a]) for a in axes])]


def enhance(enhancement_algorithm, image, step):
    image = np.pad(image, ((2, 2), (2, 2)), constant_values=(step % 2) & enhancement_algorithm[0])
    lookup_bin = np.stack([shift(image, (x, y)) for x in (-1, 0, 1) for y in (-1, 0, 1)], axis=2)
    lookup = np.dot(lookup_bin, 2**np.arange(9))
    return np.take(enhancement_algorithm, lookup)


@pytest.mark.parametrize("input_file,rounds,expected",
                         (("input/day_20_example.txt", 2, 35),
                          ("input/day_20.txt", 2, 5619),
                          ("input/day_20_example.txt", 50, 3351),
                          ("input/day_20.txt", 50, 20122)))
def test_day_20(input_file, rounds, expected):
    enhancement_algorithm, image = get_data(input_file)
    for i in range(rounds):
        image = enhance(enhancement_algorithm, image, step=i)
    result = image.sum()
    assert result == expected
