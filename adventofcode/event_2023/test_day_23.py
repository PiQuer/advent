"""
--- Day 23: A Long Walk ---
https://adventofcode.com/2023/day/23
"""
import math
from itertools import pairwise

import networkx as nx
import numpy as np
import pytest
import tinyarray as ta
from more_itertools import first
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
        while data[*current] not in (b'>', b'v') and current != self.target:
            steps += 1
            current = first(next_coordinates for h in ta_adjacent() if
                            (next_coordinates := current + h) not in seen and
                            inbounds(data.shape, next_coordinates) and
                            (((sym := data[*next_coordinates]) not in (b'#', b'>', b'v')) or h == ALLOWED[sym]))
            seen.add(current)
        if current == self.target:
            return current, steps
        return current + ALLOWED[data[*current]], steps + 1

    def get_graph(self) -> nx.DiGraph:
        result = nx.DiGraph()
        data = self.np_array_bytes
        result.add_node(self.start)
        backlog = {self.start}
        while backlog:
            if (current_node := backlog.pop()) == self.target:
                continue
            if current_node == self.start:
                next_node, steps = self.next_node(data, current_node)
                result.add_edge(current_node, next_node, weight=steps)
                backlog.add(next_node)
            else:
                for h in (b'>', b'v'):
                    start = current_node + ALLOWED[h]
                    if inbounds(data.shape, start) and data[*start] in (b'>', b'v'):
                        start += ALLOWED[h]
                        next_node, steps = self.next_node(data, start)
                        result.add_edge(current_node, next_node, weight=steps + 2)
                        backlog.add(next_node)
        return result


round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 94)], result=2190, dataset_class=DataSet)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 154)], result=6258, dataset_class=DataSet)


def single_source_longest_dag_path_length(graph: nx.DiGraph, s) -> dict[ta.ndarray_int, int]:
    assert graph.in_degree(s) == 0
    dist = dict.fromkeys(graph.nodes, -math.inf)
    dist[s] = 0
    topo_order = nx.topological_sort(graph)
    for n in topo_order:
        for source in graph.successors(n):
            if dist[source] < dist[n] + graph.edges[n, source]['weight']:
                dist[source] = dist[n] + graph.edges[n, source]['weight']
    return dist

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    graph = dataset.get_graph()
    assert is_directed_acyclic_graph(graph)
    dist = single_source_longest_dag_path_length(graph, ta.array((0, 1)))
    assert dist[dataset.target] == dataset.result

@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    graph = dataset.get_graph().to_undirected()
    simple_paths = nx.all_simple_paths(graph, dataset.start, dataset.target)
    assert max(sum(graph.edges[*e]['weight'] for e in pairwise(s)) for s in simple_paths) == dataset.result
