"""
https://adventofcode.com/2022/day/21
"""
from functools import partial
import pytest
import networkx as nx
import re
from operator import add, sub, mul, ifloordiv
from typing import Callable
from utils import dataset_parametrization, DataSetBase


def rev(operator: Callable[[int, int], int]) -> Callable[[int, int], int]:
    def rev_operator(a: int, b: int) -> int:
        return operator(b, a)
    return rev_operator


operators = {'/': ifloordiv, '*': mul, '+': add, '-': sub, None: None}
inverse = [{ifloordiv: mul, mul: ifloordiv, add: sub, sub: add},
           {ifloordiv: rev(ifloordiv), mul: ifloordiv, add: sub, sub: rev(sub)}]


class DataSet(DataSetBase):
    def graph(self) -> nx.MultiDiGraph:
        result = nx.MultiDiGraph()
        for line in self.lines():
            match = re.match(r"([a-z]{4}): ((\d+)|(([a-z]{4}) ([/*+-]) ([a-z]{4})))", line)
            result.add_node(match.group(1), operator=operators[match.group(6)])
            if match.group(5) is not None and match.group(7) is not None:
                result.add_edge(match.group(1), match.group(5))
                result.add_edge(match.group(1), match.group(7))
                result.nodes[match.group(1)]["value"] = None
            else:
                result.nodes[match.group(1)]["value"] = int(match.group(2))
        return result


round_1 = dataset_parametrization(year="2022", day="21",
                                  examples=[("", 152)], result=82225382988628, dataset_class=DataSet)
round_2 = dataset_parametrization(year="2022", day="21",
                                  examples=[("", 301)], result=3429411069028, dataset_class=DataSet)


def evaluate(node: str, graph: nx.MultiDiGraph):
    if graph.nodes[node]["operator"] is None:
        return graph.nodes[node]["value"]
    return graph.nodes[node]["operator"](*map(partial(evaluate, graph=graph), graph[node]))


def get_target(node: str, graph: nx.MultiDiGraph, value: int, ancestors: set[str], target: str = "humn"):
    if node == target:
        return value
    keys = tuple(graph[node])
    idx = 0 if keys[0] in ancestors else 1
    value = inverse[idx][graph.nodes[node]["operator"]](value, evaluate(keys[1-idx], graph))
    return get_target(keys[idx], graph, value, ancestors)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    graph = dataset.graph()
    assert evaluate('root', graph) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    graph = dataset.graph()
    graph.nodes["root"]["operator"] = sub
    assert get_target("root", graph, 0, nx.ancestors(graph, "humn") | {"humn"}) == dataset.result
