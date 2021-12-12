import pytest
from pathlib import Path
from typing import Dict, Set, List, Callable, Optional


def get_data(input_file: str) -> Dict[str, Set[str]]:
    data = [tuple(line.split('-', 2)) for line in Path(input_file).read_text().splitlines()]
    result = {}
    for a, b in data:
        if result.get(a) is None:
            result[a] = set()
        if result.get(b) is None:
            result[b] = set()
        result[a].add(b)
        result[b].add(a)
    return result


def is_loop(path: List[str], next_step: str):
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


def legal_path_part_one(path: List[str], next_step: str, visited_once: Set, _: Optional[bool]):
    if is_loop(path, next_step):
        return False
    return next_step.isupper() or next_step not in visited_once


def legal_path_part_two(path: List[str], next_step: str, visited_once: Set, visited_twice: Optional[bool]):
    if is_loop(path, next_step):
        return False
    return next_step not in visited_once or visited_twice is None


def pathfinder(path: List[str], m: Dict[str, Set[str]],
               visited_once: Set, visited_twice: Optional[str],
               legal_path: Callable[[List[str], str, Set, Optional[str]], bool]):
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


@pytest.mark.parametrize("input_file, legal_path_fn, expected", (
    ("input/day_12_example_1.txt", legal_path_part_one, 10),
    ("input/day_12_example_2.txt", legal_path_part_one, 19),
    ("input/day_12_example_3.txt", legal_path_part_one, 226),
    ("input/day_12.txt", legal_path_part_one, 4241),
    ("input/day_12_example_1.txt", legal_path_part_two, 36),
    ("input/day_12_example_2.txt", legal_path_part_two, 103),
    ("input/day_12_example_3.txt", legal_path_part_two, 3509),
    ("input/day_12.txt", legal_path_part_two, 122134),
))
def test_day_12(input_file, legal_path_fn, expected):
    data = get_data(input_file)
    paths = pathfinder(['start'], data, set(), None, legal_path_fn)
    assert len(paths) == expected
