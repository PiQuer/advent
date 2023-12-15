"""
--- Day 15: Chiton ---
https://adventofcode.com/2021/day/15
"""
import logging

import numpy as np

from utils import Waypoint, dataset_parametrization, DataSetBase, generate_rounds


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
            next_risk = best_paths[(x, y)].value + data[n]
            if n not in best_paths or best_paths[n].value > next_risk:
                best_paths[n] = Waypoint(length=length, previous=(x, y), value=next_risk)
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


round_1 = dataset_parametrization("2021", "15", [("", 40)], result=583, get_data=get_data_part_one)
round_2 = dataset_parametrization("2021", "15", [("", 315)], result=2927, get_data=get_data_part_two)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_day_15(dataset: DataSetBase):
    data = dataset.params["get_data"](dataset.input_file)
    best_paths = {(0, 0): Waypoint(previous=(0, 0), length=0, value=0)}
    length = 1
    candidates = [(0, 0)]
    while True:
        candidates = get_best_paths(length, best_paths, data, candidates)
        if not candidates:
            break
        length += 1
    result = best_paths[(data.shape[0]-1, data.shape[1]-1)].value
    logging.info("\n%s", visualize_path(data, best_paths))
    assert result == dataset.result
