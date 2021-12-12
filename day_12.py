import pytest
from pathlib import Path
from typing import Dict, Set, List
import re


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


def pathfinder(paths: List[List[str]], m: Dict[str, Set[str]]):
    result = []
    for p in paths:
        if p[-1] == 'end':
            result.append(p)
        for n in m[p[-1]]:
            if n.islower() and n in p:
                continue
            if n.isupper():
                if re.search(f"{n}(-[A-Z]+)*$", '-'.join(p)):
                    continue
            result.extend(pathfinder([p + [n]], m))
    return result


@pytest.mark.parametrize("input_file, expected", (
    ("input/day_12_example_1.txt", 10),
    ("input/day_12_example_2.txt", 19),
    ("input/day_12_example_3.txt", 226),
    ("input/day_12.txt", 4241),
))
def test_part_one(input_file, expected):
    data = get_data(input_file)
    result = len(pathfinder([['start']], data))
    assert result == expected
