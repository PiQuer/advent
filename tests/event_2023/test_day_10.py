import itertools
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pytest
import tinyarray as ta

from utils import dataset_parametrization, DataSetBase, adjacent, inbounds

pipes = {
    b"|": {ta.array((-1, 0)), ta.array((1, 0))},
    b"-": {ta.array((0, -1)), ta.array((0, 1))},
    b"L": {ta.array((-1, 0)), ta.array((0, 1))},
    b"J": {ta.array((-1, 0)), ta.array((0, -1))},
    b"7": {ta.array((1, 0)), ta.array((0, -1))},
    b"F": {ta.array((1, 0)), ta.array((0, 1))},
    b".": {},
}

outside = {
    b"|": {ta.array((1, 0)): {ta.array((0, -1))}, ta.array((-1, 0)): {ta.array((0, 1))}},
    b"-": {ta.array((0, 1)): {ta.array((1, 0))}, ta.array((0, -1)): {ta.array((-1, 0))}},
    b"L": {ta.array((1, 0)): {ta.array((0, -1)), ta.array((1, 0))}, ta.array((0, -1)): {}},
    b"J": {ta.array((1, 0)): {}, ta.array((0, 1)): {ta.array((1, 0)), ta.array((0, 1))}},
    b"7": {ta.array((0, 1)): {}, ta.array((-1, 0)): {ta.array((0, 1)), ta.array((-1, 0))}},
    b"F": {ta.array((0, -1)): {ta.array((-1, 0)), ta.array((0, -1))}, ta.array((-1, 0)): {}}
}

@dataclass
class State:
    pos: ta.ndarray_int
    next_pos: ta.ndarray_int
    field: np.ndarray
    turn: int

    def advance(self) -> "State":
        try:
            next_pos = self.next_pos + (pipes[self.field[*self.next_pos]] - {self.pos - self.next_pos}).pop()
        except KeyError:
            next_pos = None
            turn = 0
        else:
            turn = cross_product(self.pos-self.next_pos, next_pos-self.next_pos)
        return State(pos=self.next_pos, field=self.field, next_pos=next_pos, turn=turn)

class DataSet(DataSetBase):
    @cached_property
    def field(self) -> np.ndarray:
        return self.np_array_bytes()

    def starting_positions(self) -> tuple[State, State]:
        start_index = ta.array(np.argwhere(self.np_array_bytes() == b'S')[0])
        connected = list(candidate for a in adjacent() if inbounds(self.field.shape, (candidate := start_index - a))
                         and a in pipes[self.field[*candidate]])
        return State(pos=start_index, next_pos=connected[0], field=self.field,
                     turn=cross_product(connected[1]-start_index, connected[0]-start_index)), \
               State(pos=start_index, next_pos=connected[1], field=self.field,
                     turn=cross_product(connected[0]-start_index, connected[1]-start_index)),


round_1 = dataset_parametrization(year="2023", day="10", examples=[("1", 4), ("2", 8)], result=6927,
                                  dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    cursor1, cursor2 = dataset.starting_positions()
    for c in itertools.count(1):
        cursor1, cursor2 = cursor1.advance(), cursor2.advance()
        if cursor1.pos == cursor2.pos:
            break
    else:
        assert False
    assert c == dataset.result


def cross_product(v1: ta.ndarray_int, v2: ta.ndarray_int):
    return v1[0]*v2[1] - v1[1]*v2[0]


@pytest.mark.parametrize(**round_1)
def test_round_2(dataset: DataSet):
    start1, start2 = dataset.starting_positions()
    cursor = start1
    turn = cursor.turn
    while True:
        cursor = cursor.advance()
        turn += cursor.turn
        if cursor.pos == start1.pos:
            break
    start = start1 if turn == 4 else start2
