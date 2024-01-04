"""
--- Day 11: Cosmic Expansion ---
https://adventofcode.com/2023/day/11
"""
from itertools import combinations

import numpy as np

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_rounds


class DataSet(DataSetBase):
    def needs_expansion(self):
        data = self.np_array_bytes
        expand_columns = np.argwhere(np.all(data == b'.', axis=0))
        expand_rows = np.argwhere(np.all(data == b'.', axis=1))
        return expand_rows, expand_columns

    def galaxies(self, expansion: int) -> np.ndarray:
        g = np.argwhere(self.np_array_bytes == b'#')
        self._expand(g, expansion=expansion)
        return g

    def _expand(self, galaxies: np.ndarray, expansion: int):
        rows, columns = self.needs_expansion()
        galaxies[:, 0] += (expansion - 1) * np.sum(galaxies[:, 0] > rows, axis=0)
        galaxies[:, 1] += (expansion - 1) * np.sum(galaxies[:, 1] > columns, axis=0)

round_1 = dataset_parametrization(year="2023", day="11", examples=[("", 374)], dataset_class=DataSet,
                                  expansion=2, part=1)
round_2 = dataset_parametrization(year="2023", day="11", examples=[], dataset_class=DataSet,
                                  expansion=1000000, part=2)
pytest_generate_tests = generate_rounds(round_1, round_2)

def test_day_11(dataset: DataSet):
    dataset.assert_answer(sum(np.sum(np.abs(a-b))
                              for a, b in combinations(dataset.galaxies(dataset.params["expansion"]), 2)))
