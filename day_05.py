import numpy as np


def get_data(input_file: str):
    return np.fromregex(input_file, "(\d+),(\d+) -> (\d+),(\d+)", dtype=int).reshape((-1, 2, 2))


def poor_mans_line_algorithm(points, mat):
    """ Only works reliably for straight lines and diagonal lines at 45 degrees."""
    length = np.max(np.abs(points[0] - points[1])) + 1
    line = np.around(np.linspace(points[0], points[1], num=length)).astype(int)
    mat[line[:, 0], line[:, 1]] += 1


def hydrovents(input_file: str, diagonals=False):
    data = get_data(input_file)
    board_size = np.max(data) + 1
    if not diagonals:
        straight_lines = (data[:, 0, 0] == data[:, 1, 0]) | (data[:, 0, 1] == data[:, 1, 1])
        data = data[straight_lines]
    mat = np.zeros(shape=(board_size, board_size), dtype=int)
    for line_points in data:
        poor_mans_line_algorithm(line_points, mat)
    print(f"Diagonal lines are considered: {diagonals}")
    print(f"At {np.sum(mat > 1)} points at least two lines intersect.")


if __name__ == '__main__':
    hydrovents("input/day_05_example.txt", diagonals=False)
    hydrovents("input/day_05.txt", diagonals=False)
    hydrovents("input/day_05_example.txt", diagonals=True)
    hydrovents("input/day_05.txt", diagonals=True)
