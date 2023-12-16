"""
--- Day 12: Passage Pathing ---
https://adventofcode.com/2021/day/12
"""
from typing import Callable

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_rounds


class DataSet(DataSetBase):
    def get_data(self) -> dict[str, set[str]]:
        data = [tuple(line.split('-', 2)) for line in self.lines()]
        result = {}
        for a, b in data:
            if result.get(a) is None:
                result[a] = set()
            if result.get(b) is None:
                result[b] = set()
            result[a].add(b)
            result[b].add(a)
        return result


def is_loop(path: list[str], next_step: str):
    if next_step == 'start':
        return True
    if next_step.islower():
        return False
    for item in reversed(path):
        if item.islower():
            return False
        if item == next_step:
            return True
    return False


def legal_path_part_one(path: list[str], next_step: str, visited_once: set, _: bool|None):
    if is_loop(path, next_step):
        return False
    return next_step.isupper() or next_step not in visited_once


def legal_path_part_two(path: list[str], next_step: str, visited_once: set, visited_twice: bool|None):
    if is_loop(path, next_step):
        return False
    return next_step not in visited_once or visited_twice is None


def pathfinder(path: list[str], m: dict[str, set[str]],
               visited_once: set, visited_twice: str|None,
               legal_path: Callable[[list[str], str, set, str|None], bool]):
    result = []
    for n in m[path[-1]]:
        if n == 'end':
            result.append(path + [n])
            continue
        if legal_path(path, n, visited_once, visited_twice):
            next_visited_once = visited_once if n.isupper() else visited_once | {n}
            next_visited_twice = n if n in visited_once else visited_twice
            result.extend(pathfinder(path + [n], m, next_visited_once, next_visited_twice, legal_path))
    return result


round_1 = dataset_parametrization("2021", "12",
                                  [("_1", 10), ("_2", 19), ("_3", 226)], result=4241,
                                  legal_path_fn=legal_path_part_one, dataset_class=DataSet)
round_2 = dataset_parametrization("2021", "12",
                                  [("_1", 36), ("_2", 103), ("_3", 3509)], result=122134,
                                  legal_path_fn=legal_path_part_two, dataset_class=DataSet)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_day_12(dataset: DataSet):
    data = dataset.get_data()
    paths = pathfinder(['start'], data, set(), None, dataset.params["legal_path_fn"])
    assert len(paths) == dataset.result
