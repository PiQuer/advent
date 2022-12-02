import pytest
from pathlib import Path
import numpy as np


FILE_NAMES = {'example': Path("input/day_01_example.txt"), 'real': Path("input/day_01.txt")}


def _get_data(path: Path):
    current = 0
    for line in path.read_text().split('\n'):
        if not line:
            yield current
            current = 0
        else:
            current += int(line)


par = pytest.mark.parametrize("data_set,expected,top",
                              (("example", 24000, 1), ("real", 66186, 1),
                               ("example", 45000, 3), ("real", 196804, 3)))


@pytest.mark.benchmark(warmup=True, warmup_iterations=100)
class TestDay01:

    @staticmethod
    def solution_with_sort(data_set, top):
        return sum(sorted(_get_data(FILE_NAMES[data_set]))[-top:])

    @staticmethod
    def solution_with_array(data_set, top):
        data = np.fromiter(_get_data(FILE_NAMES[data_set]), dtype=np.int32)
        return np.sum(data[np.argpartition(data, -top)[-top:]])

    @staticmethod
    def solution_linear_1(data_set):
        max_calories = 0
        for calories in _get_data(FILE_NAMES[data_set]):
            if calories > max_calories:
                max_calories = calories
        return max_calories

    @staticmethod
    def solution_linear_3(data_set):
        max_0 = max_1 = max_2 = 0
        for calories in _get_data(FILE_NAMES[data_set]):
            if calories > max_2:
                max_0 = max_1
                max_1 = max_2
                max_2 = calories
            elif calories > max_1:
                max_0 = max_1
                max_1 = calories
            elif calories > max_0:
                max_0 = calories
        return max_0 + max_1 + max_2

    @par
    def test_with_sort(self, data_set, expected, top, benchmark):
        result = benchmark(self.solution_with_sort, data_set, top)
        assert result == expected

    @par
    def test_with_array(self, data_set, expected, top, benchmark):
        result = benchmark(self.solution_with_array, data_set, top)
        assert result == expected

    @par
    def test_linear(self, data_set, expected, top, benchmark):
        result = 0
        if top == 3:
            result = benchmark(self.solution_linear_3, data_set)
        if top == 1:
            result = benchmark(self.solution_linear_1, data_set)
        assert result == expected
