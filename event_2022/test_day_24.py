"""
https://adventofcode.com/2022/day/24
"""
from functools import cached_property, reduce
from itertools import chain, pairwise
from typing import Optional, Iterator
import pytest
import numpy as np
import tinyarray as ta
from dataclasses import dataclass
import heapq
from more_itertools import consume

from utils import dataset_parametrization, DataSetBase, ta_adjacent


class DataSet(DataSetBase):
    def blizzard_map(self) -> np.array:
        result = self.np_array_bytes()
        return result[1:-1, 1:-1]


round_1 = dataset_parametrization(day="24", examples=[("", 18)], result=286, dataset_class=DataSet)
round_2 = dataset_parametrization(day="24", examples=[("", 54)], result=820, dataset_class=DataSet)


@dataclass
class Item:
    pos: ta.array
    step: int
    target: ta.array

    @cached_property
    def distance(self) -> int:
        return sum(ta.abs(self.target - self.pos))

    def __hash__(self):
        return hash((self.pos, self.step))

    def __eq__(self, other):
        return self.pos == other.pos and self.step == other.step

    def __lt__(self, other: "Item"):
        return self.distance < other.distance or (self.distance == other.distance and self.step < other.step)


def is_free(pos: ta.array, board: np.array, state: int):
    return not any(True for _ in blizzards(board, pos, state))


def next_step(pq: list, start: ta.array, target: ta.array, board: np.array, seen: set[Item], shortest: Optional[int]):
    item: Item = heapq.heappop(pq)
    if shortest and item.step + sum(ta.abs(target - item.pos)) >= shortest:
        return shortest
    for next_pos in chain(ta_adjacent(item.pos), (item.pos,)):
        if next_pos == target:
            shortest = item.step + 1
            break
        if min(next_pos) < 0 or max(next_pos - board.shape) >= 0:
            if not (next_pos == item.pos and next_pos == start):  # allow to wait at the start position
                # but do not allow to enter the start position again
                continue
        if next_pos == start or is_free(next_pos, board, item.step + 1):
            next_item = Item(pos=next_pos, step=item.step+1, target=target)
            if next_item not in seen and \
                    (shortest is None or next_item.step + next_item.distance < shortest):
                seen.add(next_item)
                heapq.heappush(pq, next_item)
    return shortest


def blizzards(board: np.array, pos: ta.array, state: int) -> Iterator[str]:
    rows, columns = board.shape
    if board[(pos[0] - state) % rows, pos[1]] == b'v':
        yield 'v'
    if board[(pos[0] + state) % rows, pos[1]] == b'^':
        yield '^'
    if board[pos[0], (pos[1] - state) % columns] == b'>':
        yield '>'
    if board[pos[0], (pos[1] + state) % columns] == b'<':
        yield '<'


def visualize(board: np.array, state: int, current: Optional[ta.array] = None):
    result = np.full_like(board, '.', dtype='U1')
    rows, columns = result.shape
    for pos in np.ndindex(rows, columns):
        blist = list(blizzards(board, pos, state))
        if len(blist) == 1:
            result[pos] = blist[0]
        if len(blist) > 1:
            result[pos] = str(len(blist)) if len(blist) < 9 else '+'
    if current is not None:
        result[tuple(current)] = 'E'
    consume(map(print, (''.join(row) for row in result)))


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
