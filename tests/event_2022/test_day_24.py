"""
--- Day 24: Blizzard Basin ---
https://adventofcode.com/2022/day/24
"""
import heapq
from dataclasses import dataclass
from functools import cached_property, reduce, partial
from itertools import chain, pairwise
from operator import add, sub
from typing import Optional, Iterator

import numpy as np
import pytest
import tinyarray as ta

from utils import dataset_parametrization, DataSetBase, ta_adjacent, inbounds


class DataSet(DataSetBase):
    def blizzard_map(self) -> np.array:
        result = self.np_array_bytes()
        return result[1:-1, 1:-1]


round_1 = dataset_parametrization(year="2022", day="24", examples=[("", 18)], result=286, dataset_class=DataSet)
round_2 = dataset_parametrization(year="2022", day="24", examples=[("", 54)], result=820, dataset_class=DataSet)


@dataclass(frozen=True)
class Item:
    pos: ta.array
    step: int
    target: ta.array

    @cached_property
    def distance(self) -> int:
        return sum(ta.abs(self.target - self.pos))

    def __lt__(self, other: "Item"):
        return (self.distance, self.step) < (other.distance, other.step)


def is_free(pos: ta.array, board: np.array, state: int):
    return not any(True for _ in blizzards(board, pos, state))


def next_step(pq: list, start: ta.array, target: ta.array, board: np.array, seen: set[Item], shortest: Optional[int]):
    item: Item = heapq.heappop(pq)
    if shortest and item.step + sum(ta.abs(target - item.pos)) >= shortest:
        return shortest
    elif item.distance == 1:
        return item.step + 1
    for next_pos in chain(filter(partial(inbounds, board.shape), ta_adjacent(item.pos)), (item.pos,)):
        if (next_item := Item(pos=next_pos, step=item.step + 1, target=target)) not in seen:
            if next_item.pos == start or is_free(next_item.pos, board, next_item.step):
                seen.add(next_item)
                heapq.heappush(pq, next_item)
    return shortest


def blizzards(board: np.array, pos: ta.array, state: int) -> Iterator[bytes]:
    rows, columns = board.shape
    b = {b'v': (0, sub), b'^': (0, add), b'>': (1, sub), b'<': (1, add)}
    yield from filter(
        lambda x: board[((b[x][1](pos[0], state) % rows) if b[x][0] == 0 else pos[0],
                         (b[x][1](pos[1], state) % columns) if b[x][0] == 1 else pos[1])] == x, b)


def propagate(step: int, board: np.array, start: ta.array, target: ta.array):
    start_item = Item(pos=start, step=step, target=target)
    pq = [start_item]
    seen = {start_item}
    result = None
    while pq:
        result = next_step(pq, start, target, board, seen, result)
    return result


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    board = dataset.blizzard_map()
    start = ta.array((-1, 0))
    target = ta.array((board.shape[0], board.shape[1] - 1))
    assert propagate(0, board, start, target) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    board = dataset.blizzard_map()
    start = ta.array((-1, 0))
    target = ta.array((board.shape[0], board.shape[1] - 1))
    assert reduce(lambda x, y: propagate(x, board, *y), pairwise((start, target) * 2), 0) == dataset.result
