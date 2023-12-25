"""
--- Day 25: Snowverload ---
https://adventofcode.com/2023/day/25
"""
import operator
from functools import reduce
from itertools import starmap

import networkx as nx
import pytest
from more_itertools import consume

from adventofcode.utils import dataset_parametrization, DataSetBase

YEAR= "2023"
DAY= "25"

class DataSet(DataSetBase):
    def build_graph(self) -> nx.Graph:
        result = nx.Graph()
        for line in self.lines():
            key, values = line.split(': ', maxsplit=1)
            for value in values.split():
                result.add_edge(key, value)
        return result

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 54)], result=614655, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    graph = dataset.build_graph()
    consume(starmap(graph.remove_edge, nx.minimum_edge_cut(graph)))
    assert reduce(operator.mul, map(len, (nx.connected_components(graph))), 1) == dataset.result
