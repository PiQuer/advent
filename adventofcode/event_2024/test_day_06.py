"""
--- Day 6: Guard Gallivant ---
https://adventofcode.com/2024/day/06
"""

import networkx as nx
import numpy as np
import pytest
import tinyarray as ta
from more_itertools.more import first

from adventofcode.utils import dataset_parametrization, DataSetBase

# from adventofcode.utils import generate_parts

YEAR= "2024"
DAY= "06"

class DataSet(DataSetBase):
    def start(self) -> ta.ndarray_int:
        return ta.array(np.argwhere(self.np_array_bytes == b'^')[0])

    def graph(self) -> tuple[nx.DiGraph, tuple[ta.ndarray_int, int]]:
        m = self.np_array_bytes
        result = nx.DiGraph()
        start = (self.start() - (0, 1), 0)
        result.add_edge(start, get_edge(*start, m))
        obstacles = np.argwhere(self.np_array_bytes == b'#')
        for obstacle in obstacles:
            add_edges(m, ta.array(obstacle), result)
        return result, start

part_1 = dataset_parametrization(year=YEAR, day=DAY, part=1, dataset_class=DataSet)
part_2 = dataset_parametrization(year=YEAR, day=DAY, part=2, dataset_class=DataSet, example_results=[6])


def get_edge(position: ta.ndarray_int, orientation: int, m: np.array) -> tuple[ta.ndarray_int, int] | None:
    if position[1] == m.shape[1] - 1:
        return None
    trajectory = m[:position[0], position[1] + 1]
    obstacles = np.argwhere(trajectory == b'#').reshape(-1)
    if len(obstacles):
        return rot90((obstacles[-1], position[1] + 1), m.shape), (orientation + 1) % 4


def update_edge(m: np.array, obstacle: ta.ndarray_int, orientation: int, graph: nx.DiGraph):
    if obstacle[1] == 0 or obstacle[0] == m.shape[0]-1:
        return
    trajectory = m[obstacle[0]+1:, obstacle[1]-1]
    obstacles = np.argwhere(trajectory == b'#').reshape(-1)
    if len(obstacles):
        start_node = ta.array((obstacle[0]+1+obstacles[0], obstacle[1]-1)), orientation
        end_node = (rot90(obstacle, m.shape), (orientation+1)%4)
        if (out_edge := first(graph.out_edges(start_node), default=None)):
            graph.remove_edge(*out_edge)
        graph.add_edge(start_node, end_node)


def add_edges(m: np.array, obstacle: ta.ndarray_int, graph: nx.DiGraph, update: bool = False):
    t = m
    start_node = ta.array(obstacle)
    for orientation in range(4):
        graph.add_node((start_node, orientation))
        if (end_node := get_edge(start_node, orientation, t)) is not None:
            graph.add_edge((start_node, orientation), end_node)
        if update:
            update_edge(t, start_node, orientation, graph)
        start_node = rot90(start_node, t.shape)
        t = np.rot90(t)


def rot90(position: ta.ndarray_int, shape: tuple, k: int = 1) -> ta.ndarray_int:
    match k:
        case 1:
            return ta.array((shape[0] - 1 - position[1], position[0]))
        case -1:
            return ta.array((position[1], shape[1] - 1 - position[0]))
        case _:
            raise NotImplementedError



def get_route(m: np.array, start_node: tuple[ta.ndarray_int, int], graph: nx.DiGraph) -> bool:
    orientation = 0
    for start_node, end_node in nx.traversal.dfs_successors(graph, start_node).items():
        start = rot90(start_node[0], m.shape)
        end = end_node[0][0]
        m = np.rot90(m)
        orientation = (orientation+1) % 4
        m[start[0]-1, start[1]:end[1]:-1] = b'X'
    m[:end[0], end[1]+1] = b'X'
    while orientation % 4:
        m = np.rot90(m)
        orientation += 1

@pytest.mark.parametrize(**part_1)
def test_part_1(dataset: DataSet):
    m = dataset.np_array_bytes
    graph, start_node = dataset.graph()
    get_route(m, start_node, graph)
    dataset.assert_answer(np.sum(m == b'X'))


@pytest.mark.parametrize(**part_2)
def test_part_2(dataset: DataSet):
    m = dataset.np_array_bytes
    graph, start_node = dataset.graph()
    get_route(m, start_node, graph)
    m[*(start_node[0] + (0, 1))] = b'.'
    possible_locations = np.argwhere(m == b'X')
    result = 0
    for location in possible_locations[:100]:
        new_graph = graph.copy()
        add_edges(m, ta.array(location), new_graph, update=True)
        try:
            nx.find_cycle(new_graph, start_node)
        except nx.NetworkXNoCycle:
            continue
        else:
            result += 1
    dataset.assert_answer(result, dryrun=True)
