import pytest
from pathlib import Path
import numpy as np

deterministic_die = 100
board = 10
points_to_win = 1000


def get_start_positions(input_file):
    data = Path(input_file).read_text().splitlines()
    return tuple(int(i.split(': ')[1]) for i in data)


def play(pos_1, pos_2):
    points = np.array([0, 0])
    pos = np.array([pos_1, pos_2]) - 1
    d = 0
    winner_found = False
    while not winner_found:
        for p in (0, 1):
            pos[p] = (pos[p] + (np.arange(d, d+3) % deterministic_die + 1).sum()) % board
            points[p] += pos[p] + 1
            d += 3
            if points[p] >= points_to_win:
                winner_found = True
                break
    return d * min(points)


@pytest.mark.parametrize("input_file,expected",
                         (("input/day_21_example.txt", 739785),
                          ("input/day_21.txt", 921585)))
def test_part_one(input_file, expected):
    start_positions = get_start_positions(input_file)
    result = play(*start_positions)
    assert result == expected
