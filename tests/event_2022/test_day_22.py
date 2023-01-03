"""
https://adventofcode.com/2022/day/22
"""
from functools import partial
from itertools import takewhile, islice, cycle, product
from operator import ne
import pytest
import numpy as np
import re
import tinyarray as ta
from typing import Iterator, Optional
from more_itertools import consume, last, lstrip, first
from dataclasses import dataclass
import networkx as nx
from utils import dataset_parametrization, DataSetBase, ta_directions_arrows


def visualize(c: np.array, pos: Optional[ta.array] = None):
    to_visualize = c[0]
    if pos is not None:
        to_visualize = to_visualize.copy()
        to_visualize[tuple(pos)] = 2
    to_visualize = np.vectorize(inv_byte_map.get)(to_visualize)
    consume(map(print, (''.join(row) for row in to_visualize)))


@dataclass
class EdgeWrap:
    from_c: list[ta.array]
    to_c: list[ta.array]
    from_facing: str
    to_facing: str


byte_map = {' ': -1, '#': 0, '.': 1, 'P': 2}
inv_byte_map = {v: k for k, v in byte_map.items()}
facing_value_map = {'>': 0, 'v': 1, '<': 2, '^': 3}
inv_facing_map = {'>': '<', '<': '>', 'v': '^', '^': 'v'}
rot_facing_map = {
    'R': {'>': 'v', 'v': '<', '<': '^', '^': '>'},
    'L': {'>': '^', '^': '<', '<': 'v', 'v': '>'},
}


def rotate(m: np.array, direction: str, pos: ta.array, facing: int) -> tuple[np.array, ta.array, int]:
    rotated = m
    if direction == "R":
        pos = ta.array((m.shape[2] - 1 - pos[1], pos[0]))
        rotated = np.rot90(m, axes=(1, 2))
        facing += 1
    elif direction == "L":
        pos = ta.array((pos[1], m.shape[1] - 1 - pos[0]))
        rotated = np.rot90(m, k=-1, axes=(1, 2))
        facing -= 1
    assert rotated.base is m or rotated.base is m.base or rotated is m
    return rotated, pos, facing % 4


class DataSet(DataSetBase):
    def board(self) -> tuple[np.array, ta.array]:
        _map = self.separated_by_empty_line()[0].split("\n")
        _max = max(map(len, _map))
        x = np.array([list(map(byte_map.get, row + " "*(_max - len(row)))) for row in _map], dtype=int)
        x = np.concatenate((x[np.newaxis, :], np.indices(x.shape)))
        return x, ta.array((0, next(np.nditer(np.argwhere(x[0, 0] == 1)))))

    def instructions(self) -> Iterator[tuple[str, int]]:
        i_str = "S" + self.separated_by_empty_line()[1]
        return map(lambda m: (m.group(1), int(m.group(2))), re.finditer(r"([SRL])(\d+)", i_str))

    def graph(self, edge_wraps: tuple[EdgeWrap]) -> tuple[nx.MultiDiGraph, ta.array]:
        board, pos = self.board()
        result = nx.MultiDiGraph()
        directions = ta_directions_arrows()
        for node in map(ta.array, iter(board[1:3, board[0] == 1].transpose())):
            result.add_node(node)
            for d in directions:
                new_pos = node + directions[d]
                if 0 <= min(new_pos) and max(new_pos - board[0].shape) <= -1 and board[0][tuple(new_pos)] == 1:
                    result.add_edge(node, node + directions[d], **{d: d})
            for ew in edge_wraps:
                if node in ew.from_c:
                    new_node = ew.to_c[ew.from_c.index(node)]
                    if board[0][tuple(new_node)] == 1:
                        result.add_edge(node, new_node, **{ew.from_facing: ew.to_facing})
                    break
                elif node in ew.to_c:
                    new_node = ew.from_c[ew.to_c.index(node)]
                    if board[0][tuple(new_node)] == 1:
                        result.add_edge(node, ew.from_c[ew.to_c.index(node)],
                                        **{inv_facing_map[ew.to_facing]: inv_facing_map[ew.from_facing]})
                    break
        return result, pos


def walk_round_1(m: np.array, pos: ta.array, instr: tuple[str, int], facing: int) -> tuple[np.array, ta.array, int]:
    m, pos, facing = rotate(m, instr[0], pos, facing)
    row = m[0, pos[0]]
    indices = np.arange(len(row))
    start = indices[pos[1]]
    pos = ta.array((pos[0],
                    last(takewhile(row.__getitem__, islice(lstrip(cycle(np.nditer(indices[np.where(row >= 0)])),
                                                                  partial(ne, start)), instr[1] + 1)))))
    return m, pos, facing


round_1 = dataset_parametrization(year="2022", day="22", examples=[("", 6032)], result=30552, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    board, pos = dataset.board()
    facing = 0
    for instr in dataset.instructions():
        board, pos, facing = walk_round_1(board, pos, instr, facing)
    assert 1000 * (pos[0] + 1) + 4 * (pos[1] + 1) + facing == dataset.result


puzzle_edge_wraps = {
    "example": (
        EdgeWrap(list(map(ta.array, product((0,), range(8, 12)))),
                 list(map(ta.array, product((4,), reversed(range(0, 4))))), "^", "v"),
        EdgeWrap(list(map(ta.array, product(range(0, 4), (8,)))),
                 list(map(ta.array, product((4,), range(4, 8)))), "<", "v"),
        EdgeWrap(list(map(ta.array, product(range(0, 4), (11,)))),
                 list(map(ta.array, product(reversed(range(8, 12)), (15,)))), ">", "<"),
        EdgeWrap(list(map(ta.array, product(range(4, 8), (0,)))),
                 list(map(ta.array, product((11,), reversed(range(12, 16))))), "<", "^"),
        EdgeWrap(list(map(ta.array, product(range(4, 8), (11,)))),
                 list(map(ta.array, product((8,), reversed(range(12, 16))))), ">", "v"),
        EdgeWrap(list(map(ta.array, product((7,), range(0, 4)))),
                 list(map(ta.array, product((11,), reversed(range(8, 12))))), "v", "^"),
        EdgeWrap(list(map(ta.array, product((7,), range(4, 8)))),
                 list(map(ta.array, product(reversed(range(8, 12)), (8,)))), "v", ">")
    ),
    "puzzle": (
        EdgeWrap(list(map(ta.array, product((0,), range(50, 100)))),
                 list(map(ta.array, product(range(150, 200), (0,)))), "^", ">"),
        EdgeWrap(list(map(ta.array, product(range(0, 50), (50,)))),
                 list(map(ta.array, product(reversed(range(100, 150)), (0,)))), "<", ">"),
        EdgeWrap(list(map(ta.array, product(range(0, 50), (149,)))),
                 list(map(ta.array, product(reversed(range(100, 150)), (99,)))), ">", "<"),
        EdgeWrap(list(map(ta.array, product(range(50, 100), (50,)))),
                 list(map(ta.array, product((100,), range(0, 50)))), "<", "v"),
        EdgeWrap(list(map(ta.array, product(range(50, 100), (99,)))),
                 list(map(ta.array, product((49,), range(100, 150)))), ">", "^"),
        EdgeWrap(list(map(ta.array, product(range(150, 200), (49,)))),
                 list(map(ta.array, product((149,), range(50, 100)))), ">", "^"),
        EdgeWrap(list(map(ta.array, product((0,), range(100, 150)))),
                 list(map(ta.array, product((199,), range(0, 50)))), "^", "^")
    )
}


round_2 = dataset_parametrization(year="2022", day="22",
                                  examples=[("", 5031, {'edge_wrap': puzzle_edge_wraps["example"]})],
                                  result=184106, dataset_class=DataSet, edge_wrap=puzzle_edge_wraps["puzzle"])


def step_iterator(pos: ta.array, graph: nx.MultiDiGraph, facing: str) -> Iterator[ta.array]:
    while pos:
        av = graph[pos]
        pos = first((edge for edge in av if facing in av[edge][0]), None)
        if pos:
            facing = av[pos][0][facing]
            yield pos, facing


def walk_round_2(pos: ta.array, graph: nx.MultiDiGraph, instr: tuple[str, int], facing: str):
    if instr[0] in rot_facing_map:
        facing = rot_facing_map[instr[0]][facing]
    return last(islice(step_iterator(pos, graph, facing), instr[1]), (pos, facing))


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    board, _ = dataset.board()
    graph, pos = dataset.graph(dataset.params["edge_wrap"])
    facing = ">"
    for instr in dataset.instructions():
        pos, facing = walk_round_2(pos, graph, instr, facing)
    assert 1000 * (pos[0] + 1) + 4 * (pos[1] + 1) + facing_value_map[facing] == dataset.result
