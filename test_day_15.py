import pytest
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Waypoint:
    length: int
    previous: Tuple[int, int]
    risk: int


def get_data_part_one(input_file):
    return np.genfromtxt(input_file, dtype=int, delimiter=1)


def get_data_part_two(input_file):
    data_patch = get_data_part_one(input_file) - 1
    data = data_patch
    for col in range(1, 5):
        data = np.concatenate((data, data_patch+col), axis=1)
    data_patch = data
    for row in range(1, 5):
        data = np.concatenate((data, data_patch+row), axis=0)
    data %= 9
    return data + 1


def get_best_paths(length, best_paths, data, candidates):
    next_candidates = []
    for x, y in candidates:
        next_points = (_next for _next in ((x+1, y), (x, y+1), (x-1, y), (x, y-1))
                       if 0 <= _next[0] < data.shape[0] and 0 <= _next[1] < data.shape[1])
        for n in next_points:
            next_risk = best_paths[(x, y)].risk + data[n]
            if n not in best_paths or best_paths[n].risk > next_risk:
                best_paths[n] = Waypoint(length=length, previous=(x, y), risk=next_risk)
                next_candidates.append(n)
    return next_candidates


def visualize_path(data, best_path):
    visualize = data.copy()
    next_point = (data.shape[0]-1, data.shape[1]-1)
    visualize[next_point] = 100 + data[next_point]
    while not next_point == (0, 0):
        next_point = best_path[next_point].previous
        visualize[next_point] = 100 + data[next_point]
    return visualize


@pytest.mark.parametrize("input_file,data_fn,expected",
                         (("input/day_15_example.txt", get_data_part_one, 40),
                          ("input/day_15.txt", get_data_part_one,  583),
                          ("input/day_15_example.txt", get_data_part_two, 315),
                          ("input/day_15.txt", get_data_part_two, 2927)))
def test_day_15(input_file, data_fn, expected):
    data = data_fn(input_file)
    best_paths = {(0, 0): Waypoint(previous=(0, 0), length=0, risk=0)}
    length = 1
    candidates = [(0, 0)]
    while True:
        candidates = get_best_paths(length, best_paths, data, candidates)
        if not candidates:
            break
        length += 1
    result = best_paths[(data.shape[0]-1, data.shape[1]-1)].risk
    _path = visualize_path(data, best_paths)
    assert result == expected
