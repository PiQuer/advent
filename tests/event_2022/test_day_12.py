"""
--- Day 12: Hill Climbing Algorithm ---
https://adventofcode.com/2022/day/12
"""
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import starmap, product, takewhile
from operator import add
from typing import Iterable, Union

import numpy as np
import pytest
import tinyarray as ta
from more_itertools import one, peekable, iterate, chunked

from utils import dataset_parametrization, DataSetBase, ta_adjacent, Waypoint

round_1 = dataset_parametrization(year="2022", day="12", examples=[("", 31)], result=380)
round_2 = dataset_parametrization(year="2022", day="12", examples=[("", 29)], result=375)


adjacent = tuple(ta_adjacent())


@dataclass
class ShortestPath:
    length: Union[float, int]
    path: list[ta.array]


def next_candidates(length: int, candidates: Iterable[ta.array], seen: defaultdict[ta.array, Waypoint],
                    heightmap: ta.array, target: ta.array):
    def valid_next(arg: tuple[ta.array, ta.array]):
        direction, pos = arg
        new_pos = direction + pos
        if min(new_pos) < 0 or min(heightmap.shape - new_pos) <= 0 or seen[new_pos].length <= length \
                or heightmap[new_pos] - heightmap[pos] > 1 or seen[target].length <= length:
            return False
        seen[new_pos] = Waypoint(length=length, previous=pos)
        return True
    return peekable(starmap(add, filter(valid_next, product(adjacent, candidates))))


def path_finder(heightmap: ta.array, start: ta.array, target: ta.array) \
        -> ShortestPath:
    seen = defaultdict(lambda: Waypoint(length=math.inf, previous=None))
    seen[start] = Waypoint(length=0, previous=None)
    candidates = peekable([start])
    length = 1
    while (candidates := next_candidates(length, candidates, seen, heightmap, target)).peek(False):
        length += 1
    return ShortestPath(
        seen[target].length,
        list(takewhile(lambda x: x is not None, iterate(lambda x: seen[x].previous, seen[target].previous)))
    )


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    heightmap_np = dataset.np_array_bytes()
    start = ta.array(one(zip(*np.where(heightmap_np == b'S'))))
    target = ta.array(one(zip(*np.where(heightmap_np == b'E'))))
    heightmap_np[tuple(start)] = b'a'
    heightmap_np[tuple(target)] = b'z'
    heightmap = ta.array(heightmap_np.view(np.int8) - ord(b'a'), int)
    result = path_finder(heightmap, start, target)
    print()
    heightmap_np[tuple(zip(*result.path))] = b' '
    print('\n'.join(''.join(line) for line in chunked(heightmap_np.tobytes().decode(), heightmap_np.shape[1])))
    assert result.length == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    heightmap_np = dataset.np_array_bytes()
    start_s = ta.array(one(zip(*np.where(heightmap_np == b'S'))))
    target = ta.array(one(zip(*np.where(heightmap_np == b'E'))))
    heightmap_np[tuple(start_s)] = b'a'
    heightmap_np[tuple(target)] = b'z'
    start = map(ta.array, zip(*np.where(heightmap_np == b'a')))
    heightmap = ta.array(heightmap_np.view(np.int8) - ord(b'a'), int)
    result = min((path_finder(heightmap, s, target) for s in start), key=lambda x: x.length)
    print()
    heightmap_np[tuple(zip(*result.path))] = b' '
    print('\n'.join(''.join(line) for line in chunked(heightmap_np.tobytes().decode(), heightmap_np.shape[1])))
    assert result.length == dataset.result
