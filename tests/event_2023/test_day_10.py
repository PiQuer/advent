import itertools
from dataclasses import dataclass
from functools import cached_property
from itertools import takewhile

import numpy as np
import pytest
import tinyarray as ta
from more_itertools import iterate, one

from utils import dataset_parametrization, DataSetBase, adjacent, inbounds

rotation = ta.array([[0, 1], [-1, 0]])

pipes = {
    b"|": {ta.array((-1, 0)), ta.array((1, 0))},
    b"-": {ta.array((0, -1)), ta.array((0, 1))},
    b"L": {ta.array((-1, 0)), ta.array((0, 1))},
    b"J": {ta.array((-1, 0)), ta.array((0, -1))},
    b"7": {ta.array((1, 0)), ta.array((0, -1))},
    b"F": {ta.array((1, 0)), ta.array((0, 1))},
    b".": {},
}


@dataclass
class State:
    pos: ta.ndarray_int
    next_pos: ta.ndarray_int
    prev_pos: ta.ndarray_int

    @cached_property
    def turn(self) -> int:
        return cross_product((self.next_pos - self.pos), (self.prev_pos - self.pos))

    @cached_property
    def inside_neighbors(self) -> set["State"]:
        v1 = self.prev_pos - self.pos
        v2 = self.next_pos - self.pos
        return set(self.pos + z for z in
                   takewhile(lambda v: v != v2, iterate(lambda x: ta.dot(rotation, x), ta.dot(rotation, v1))))


def advance(s: State, field: np.ndarray) -> "State":
    next_pos = s.next_pos + one(pipes[field[*s.next_pos]] - {s.pos - s.next_pos})
    return State(pos=s.next_pos, next_pos=next_pos, prev_pos=s.pos)


class DataSet(DataSetBase):
    @cached_property
    def field(self) -> np.ndarray:
        return self.np_array_bytes()

    def starting_positions(self) -> tuple[State, State]:
        start_index = ta.array(np.argwhere(self.np_array_bytes() == b'S')[0])
        connected = list(candidate for a in adjacent() if inbounds(self.field.shape, (candidate := start_index - a))
                         and a in pipes[self.field[*candidate]])
        return State(pos=start_index, next_pos=connected[0], prev_pos=connected[1]), \
               State(pos=start_index, next_pos=connected[1], prev_pos=connected[0])


round_1 = dataset_parametrization(year="2023", day="10", examples=[("1", 4), ("2", 8)], result=6927,
                                  dataset_class=DataSet)
round_2 = dataset_parametrization(year="2023", day="10", examples=[("3", 4), ("4", 8), ("5", 10)], result=467,
                                  dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    cursor1, cursor2 = dataset.starting_positions()
    for c in itertools.count(1):
        cursor1, cursor2 = advance(cursor1, dataset.field), advance(cursor2, dataset.field)
        if cursor1.pos == cursor2.pos:
            break
    else:
        assert False
    assert c == dataset.result


def cross_product(v1: ta.ndarray_int, v2: ta.ndarray_int):
    return v1[0]*v2[1] - v1[1]*v2[0]


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    start1, start2 = dataset.starting_positions()
    cursor1, cursor2 = start1, start2
    loop1, loop2 = {}, {}
    turn1, turn2 = 0, 0
    while True:
        turn1 += cursor1.turn
        turn2 += cursor2.turn
        loop1[cursor1.pos] = cursor1
        loop2[cursor2.pos] = cursor2
        if cursor1.next_pos == start1.pos:
            break
        cursor1 = advance(cursor1, dataset.field)
        cursor2 = advance(cursor2, dataset.field)
    loop = loop1 if turn1 == 4 else loop2
    inside_tiles = set(t for l in loop.values() for t in l.inside_neighbors if t not in loop)
    to_examine = inside_tiles.copy()
    while to_examine:
        next_tile = to_examine.pop()
        for a in adjacent():
            if (candidate := next_tile + a) not in inside_tiles and candidate not in loop:
                inside_tiles.add(candidate)
                to_examine.add(candidate)
    assert len(inside_tiles) == dataset.result