"""
--- Day 9: Mirage Maintenance ---
https://adventofcode.com/2023/day/9
"""
from functools import cached_property

import numpy as np

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_parts


class DataSet(DataSetBase):
    @cached_property
    def data(self):
        return self.np_array(dtype=np.float64).transpose()

    @cached_property
    def x_data(self):
        return np.expand_dims(np.arange(self.data.shape[0]), axis=1)


part1 = dataset_parametrization(year="2023", day="09", examples=[("", 114, {'idx': 0})], part=1, dataset_class=DataSet,
                                idx=0)
part2 = dataset_parametrization(year="2023", day="09", examples=[("", 2, {'idx': 1})], part=2, dataset_class=DataSet,
                                idx=1)
pytest_generate_tests = generate_parts(part1, part2)

def poly_newton_coefficients(x, y):
    """
    x: list or np array containing x data points
    y: list or np array containing y data points
    """
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])
    return a


def newton_polynomial(x_data, a, x):
    """
    x_data: data points at x
    a: newton coefficients
    x: evaluation point(s)
    """
    n = len(x_data) - 1  # Degree of polynomial
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k])*p
    return p


def test_day_09(dataset: DataSet):
    x = np.array([[[dataset.x_data[-1, 0] + 1, -1]]])
    coefficients = poly_newton_coefficients(dataset.x_data, dataset.data)
    results = np.round(newton_polynomial(np.expand_dims(dataset.x_data, axis=2),
                                         np.expand_dims(coefficients, axis=2), x))
    dataset.assert_answer(np.sum(results.astype(np.int32), axis=1).reshape(2,)[dataset.params['idx']])
