"""
--- Day 20: Trench Map ---
https://adventofcode.com/2021/day/20
"""
import numpy as np
import tinyarray as ta

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_rounds


class DataSet(DataSetBase):
    # noinspection PyUnresolvedReferences
    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.separated_by_empty_line()
        enhancement_algorithm = (np.array(list(data[0])) == '#').astype(int)
        image = (np.array([list(line) for line in data[1].splitlines()]) == '#').astype(int)
        return enhancement_algorithm, image


def shift(array: np.array, axes):
    slices = {-1: (2, None), 0: (1, -1), 1: (None, -2)}
    return array[*(slice(*slices[a]) for a in axes)]


def enhance(enhancement_algorithm: np.ndarray, image: np.ndarray, step):
    image = np.pad(image, ta.array(((2, 2), (2, 2))), constant_values=(step % 2) & enhancement_algorithm[0])
    lookup_bin = np.stack([shift(image, (x, y)) for x in (-1, 0, 1) for y in (-1, 0, 1)], axis=2)
    lookup = np.dot(lookup_bin, 2**np.arange(9))
    return np.take(enhancement_algorithm, lookup)


round_1 = dataset_parametrization("2021", "20", [("", 35)], result=5619, rounds=2,
                                  dataset_class=DataSet)
round_2 = dataset_parametrization("2021", "20", [("", 3351)], result=20122, rounds=50,
                                  dataset_class=DataSet)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_day_20(dataset: DataSet):
    enhancement_algorithm, image = dataset.get_data()
    for i in range(dataset.params["rounds"]):
        image = enhance(enhancement_algorithm, image, step=i)
    assert image.sum() == dataset.result
