import numpy as np
from numpy.lib import recfunctions as rfn
import pytest


def get_data(input_file: str):
    data_type = np.dtype(','.join(['int32'] * 4))
    data = np.fromregex(input_file, r"(\d+),(\d+) -> (\d+),(\d+)", dtype=data_type)
    return rfn.structured_to_unstructured(data).reshape((-1, 2, 2))


def poor_mans_line_algorithm(points, mat):
    """ Only works reliably for straight lines and diagonal lines at 45 degrees."""
    length = np.max(np.abs(points[0] - points[1])) + 1
    line = np.around(np.linspace(points[0], points[1], num=length)).astype(int)
    mat[line[:, 0], line[:, 1]] += 1


@pytest.mark.parametrize("input_file", ["input/day_05_example.txt", "input/day_05.txt"])
@pytest.mark.parametrize("diagonals", [False, True])
def test_hydrovents(input_file: str, diagonals):
    data = get_data(input_file)
    board_size = np.max(data) + 1
    if not diagonals:
        straight_lines = (data[:, 0, 0] == data[:, 1, 0]) | (data[:, 0, 1] == data[:, 1, 1])
        data = data[straight_lines]
    mat = np.zeros(shape=(board_size, board_size), dtype=int)
    for line_points in data:
        poor_mans_line_algorithm(line_points, mat)
    result = np.sum(mat > 1)
    print(f"At {result} points at least two lines intersect.")
    example = "example" in input_file
    assert result == \
           {(True, False): 5, (True, True): 12, (False, False): 6225, (False, True): 22116}[(example, diagonals)]
