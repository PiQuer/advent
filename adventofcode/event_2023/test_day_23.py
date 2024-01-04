"""
--- Day 23: A Long Walk ---
https://adventofcode.com/2023/day/23
"""
from functools import partial
from itertools import pairwise
from multiprocessing import Pool

import igraph as ig
import networkx as nx
import numpy as np
import pytest
import tinyarray as ta
from more_itertools import first, one
from networkx import is_directed_acyclic_graph

from adventofcode.utils import dataset_parametrization, DataSetBase, ta_adjacent, inbounds

YEAR= "2023"
DAY= "23"


ALLOWED = {
    b">": ta.array((0, 1)),
    b"v": ta.array((1, 0)),
    b"#": ta.array((0, 0))
}

class DataSet(DataSetBase):
    @property
    def start(self) -> ta.ndarray_int:
        return ta.array((0, 1))

    @property
    def target(self) -> ta.ndarray_int:
        return ta.array(self.np_array_bytes.shape) - (1, 2)

    def next_node(self, data: np.ndarray, start: ta.ndarray_int) -> tuple[ta.ndarray_int, int]:
        seen = {start}
        current = start
        steps = 0
        while data[(*current,)] not in (b'>', b'v') and current != self.target:
            steps += 1
            current = first(next_coordinates for h in ta_adjacent() if
                            (next_coordinates := current + h) not in seen and
                            inbounds(data.shape, next_coordinates) and
                            (((sym := data[(*next_coordinates,)]) not in (b'#', b'>', b'v')) or h == ALLOWED[sym]))
            seen.add(current)
        if current == self.target:
            return current, steps
        return current + ALLOWED[data[(*current,)]], steps + 1

    def get_graph(self) -> nx.DiGraph:
        result = nx.DiGraph()
        data = self.np_array_bytes
        result.add_node(str(self.start))
        backlog = {self.start}
        while backlog:
            if (current_node := backlog.pop()) == self.target:
                continue
            if current_node == self.start:
                next_node, steps = self.next_node(data, current_node)
                result.add_edge(str(current_node), str(next_node), weight=steps)
                backlog.add(next_node)
            else:
                for h in (b'>', b'v'):
                    start = current_node + ALLOWED[h]
                    if inbounds(data.shape, start) and data[(*start,)] != b'#':
                        start += ALLOWED[h]
                        next_node, steps = self.next_node(data, start)
                        result.add_edge(str(current_node), str(next_node), weight=steps + 2)
                        backlog.add(next_node)
        return result


round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 94)], dataset_class=DataSet, part=1)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 154)], dataset_class=DataSet, part=2)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    graph = dataset.get_graph()
    assert is_directed_acyclic_graph(graph)
    dataset.assert_answer(nx.dag_longest_path_length(graph, weight="weight"))


def length(path, weights):
    return sum(weights[e] for e in pairwise(path))


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    graph = dataset.get_graph().to_undirected()
    last_to_exit = one(graph[str(dataset.target)].keys())
    h = ig.Graph.from_networkx(graph)
    h.vs["name"] = h.vs["_nx_name"]
    weights = {}
    for edge in h.es:
        nodes = (edge.source, edge.target)
        nodes_reversed = (edge.target, edge.source)
        weights[nodes] = edge['weight']
        weights[nodes_reversed] = edge['weight']
    simple_paths = h.get_all_simple_paths(h.vs.find(str(dataset.start)), to=h.vs.find(last_to_exit))
    with Pool(processes=None) as pool:
        dist = max(pool.map(partial(length, weights=weights), simple_paths))
    dataset.assert_answer(dist + graph.edges[last_to_exit, str(dataset.target)]['weight'])
