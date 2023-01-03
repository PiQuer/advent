import numpy as np
from numpy.lib import recfunctions as rfn

from utils import dataset_parametrization, DataSetBase, generate_rounds


class DataSet(DataSetBase):
    def get_data(self):
        data_type = np.dtype(','.join(['int32'] * 4))
        data = np.fromregex(self.input_file, r"(\d+),(\d+) -> (\d+),(\d+)", dtype=data_type)
        return rfn.structured_to_unstructured(data).reshape((-1, 2, 2))


def poor_mans_line_algorithm(points, mat):
    """ Only works reliably for straight lines and diagonal lines at 45 degrees."""
    length = np.max(np.abs(points[0] - points[1])) + 1
    line = np.around(np.linspace(points[0], points[1], num=length)).astype(int)
    mat[line[:, 0], line[:, 1]] += 1


round_1 = dataset_parametrization(year="2021", day="05", examples=[("", 5)], result=6225, diagonals=False,
                                  dataset_class=DataSet)
round_2 = dataset_parametrization(year="2021", day="05", examples=[("", 12)], result=22116, diagonals=True,
                                  dataset_class=DataSet)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_hydrovents(dataset: DataSet):
    data = dataset.get_data()
    diagonals = dataset.params["diagonals"]
    board_size = np.max(data) + 1
    if not diagonals:
        straight_lines = (data[:, 0, 0] == data[:, 1, 0]) | (data[:, 0, 1] == data[:, 1, 1])
        data = data[straight_lines]
    mat = np.zeros(shape=(board_size, board_size), dtype=int)
    for line_points in data:
        poor_mans_line_algorithm(line_points, mat)
    assert np.sum(mat > 1) == dataset.result
