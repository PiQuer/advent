"""
https://adventofcode.com/2022/day/14
"""
from itertools import product, repeat, takewhile
from more_itertools import iterate, pairwise
from typing import Optional
import pytest
import tinyarray as ta
from collections import defaultdict
from operator import methodcaller, add
from utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def build_map(self):
        cave = defaultdict(lambda: '.')
        for line in self.lines():
            for a, b in pairwise(map(methodcaller("split", ","), line.split(" -> "))):
                a = ta.array(tuple(map(int, a)))
                b = ta.array(tuple(map(int, b)))
                for x, y in product(range(min(a[0], b[0]), max(a[0], b[0]) + 1),
                                    range(min(a[1], b[1]), max(a[1], b[1]) + 1)):
                    cave[ta.array((x, y))] = '#'
        return cave, max(cave, key=lambda z: z[1])[1]


class EndOfProblem(Exception):
    pass


round_1 = dataset_parametrization(year="2022", day="14", examples=[("", 24)], result=805, dataset_class=DataSet)
round_2 = dataset_parametrization(year="2022", day="14", examples=[("", 93)], result=25161, dataset_class=DataSet)


def find_path(start: ta.array, cave: defaultdict, max_depth: int, endless: bool = True):
    def next_pos(pos: ta.array) -> Optional[ta.array]:
        if pos[1] >= max_depth:
            if endless:
                raise EndOfProblem()
            return None
        down, left, right = map(add, (ta.array((x, 1)) for x in (0, -1, 1)), repeat(pos))
        return down if cave[down] == '.' else (left if cave[left] == '.' else (right if cave[right] == '.' else None))
    return list(takewhile(lambda x: x, iterate(next_pos, start)))


def simulate_flow(cave, max_depth, endless=True):
    to_check = [ta.array((500, 0))]
    while to_check:
        cave[(path := find_path(to_check[-1], cave, max_depth, endless=endless))[-1]] = 'O'
        to_check.pop()
        to_check += path[:-1]


def visualize_cave(cave, max_depth, endless=True):
    min_col = min(cave, key=lambda x: x[0])[0]
    max_col = max(cave, key=lambda x: x[0])[0]
    print()
    for line in range(max_depth + 1):
        print(''.join(cave[ta.array((x, line))] for x in range(min_col, max_col + 1)))
    if not endless:
        print('#' * (max_col - min_col + 1))


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    cave, max_depth = dataset.build_map()
    try:
        simulate_flow(cave, max_depth)
    except EndOfProblem:
        pass
    visualize_cave(cave, max_depth)
    assert sum(1 for key in cave if cave[key] == 'O') == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    cave, max_depth = dataset.build_map()
    max_depth += 1
    simulate_flow(cave, max_depth, endless=False)
    visualize_cave(cave, max_depth, endless=False)
    assert sum(1 for key in cave if cave[key] == 'O') == dataset.result
