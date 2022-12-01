import pytest
from pathlib import Path
import numpy as np


def _get_data(path: Path):
    current = 0
    calories = []
    for line in path.read_text().split('\n'):
        if not line:
            calories.append(current)
            current = 0
        else:
            current += int(line)
    return calories


@pytest.fixture(scope="module")
def data():
    return {
        'example': _get_data(Path("input/day_01_example.txt")),
        'real': _get_data(Path("input/day_01.txt"))
    }


par = pytest.mark.parametrize("data_set,expected,top",
                              (("example", 24000, 1), ("real", 66186, 1),
                               ("example", 45000, 3), ("real", 196804, 3)))


class TestDay01:
    @pytest.fixture(scope="class")
    def data_sorted(self, data):
        return {
            'example': list(sorted(data['example'])),
            'real': list(sorted(data['real']))
        }

    @pytest.fixture(scope="class")
    def data_array(self, data):
        return {
            'example': np.array(data['example']),
            'real': np.array(data['real'])
        }

    @par
    def test_with_sort(self, data_sorted, data_set, expected, top):
        assert sum(data_sorted[data_set][-top:]) == expected

    @par
    def test_with_array(self, data_array, data_set, expected, top):
        data = data_array[data_set]
        assert np.sum(data[np.argpartition(data, -top)[-top:]]) == expected
