"""
https://adventofcode.com/2022/day/16
"""
import pytest
import networkx as nx
import re
from typing import Iterable, Union
from dataclasses import dataclass
from itertools import product
from utils import dataset_parametrization, DataSetBase


@dataclass
class State:
    minute: int
    pressure: int
    pos: Union[str, tuple[str, str]]
    opened: Union[tuple[str], tuple[()]]
    seen: tuple[str]

    def __hash__(self):
        return hash((self.pos, self.opened))

    def __eq__(self, other: "State"):
        return self.pos == other.pos and self.opened == other.opened


VALVES = set()


class DataSet(DataSetBase):
    def build_graph(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for line in self.lines():
            match = re.match(r"Valve ([A-Z]{2}) has flow rate=(\d+);.*valves? ([A-Z, ]*)", line)
            node, flow_rate = match.groups()[:2]
            targets = match.group(3).split(', ')
            graph.add_edges_from((node, target, {"flow_rate": 0}) for target in targets)
            if flow_rate:
                graph.add_edge(node, node, flow_rate=int(flow_rate))
                VALVES.add(node)
        return graph


round_1 = dataset_parametrization(day="16", examples=[("", 1651)], result=1559, dataset_class=DataSet)
round_2 = dataset_parametrization(day="16", examples=[("", 1707)], result=None, dataset_class=DataSet)


def get_next_candidates(graph: nx.MultiDiGraph, candidates: Iterable[State],
                        seen: dict[State, int]) -> Iterable[State]:
    result = []
    for c in candidates:
        if len(c.opened) == len(VALVES) or not c.minute:
            continue
        next_minute = c.minute - 1
        for nbr in graph[c.pos]:
            nbr: str
            edge = graph[c.pos][nbr][0]
            if (fr := edge['flow_rate']) and nbr not in c.opened:
                candidate = State(minute=next_minute, pressure=c.pressure + fr * next_minute, pos=nbr,
                                  opened=tuple(sorted(c.opened + (nbr,))), seen=c.seen)
            elif nbr != c.pos:
                candidate = State(minute=next_minute, pressure=c.pressure, pos=nbr, opened=c.opened,
                                  seen=tuple(sorted(c.seen + (nbr,))))
            else:
                continue
            if seen.get(candidate, -1) < candidate.pressure:
                result.append(candidate)
                seen[candidate] = candidate.pressure
    return result


def get_next_candidates2(graph: nx.MultiDiGraph, candidates: Iterable[State], seen: dict[State, int]) \
        -> Iterable[State]:
    result = []
    for c in candidates:
        _openend, _seen, _pressure = c.opened, c.seen, c.pressure
        if len(_openend) == len(VALVES) or not c.minute:
            continue
        next_minute = c.minute - 1
        for n1, n2 in product(graph[c.pos[0]], graph[c.pos[1]]):
            e1, e2 = graph[c.pos[0]][n1][0], graph[c.pos[1]][n2][0]
            if e1 == e2 and e1['flow_rate']:
                continue
            for nbr, e, pos in zip((n1, n2), (e1, e2), c.pos):
                if (fr := e['flow_rate']) and nbr not in _openend:
                    _openend = tuple(sorted(_openend + (nbr,)))
                    _pressure += fr * next_minute
                elif nbr != pos:
                    _seen = tuple(sorted(_seen + (nbr,)))
                else:
                    continue
            candidate = State(minute=next_minute, pressure=_pressure, pos=(n1, n2), opened=_openend, seen=_seen)
            if seen.get(candidate, -1) < candidate.pressure:
                result.append(candidate)
                seen[candidate] = candidate.pressure
        return result


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    graph = dataset.build_graph()
    candidates = [State(minute=30, pos='AA', pressure=0, opened=(), seen=('AA',))]
    seen = {candidates[0]: 0}
    while candidates:
        candidates = get_next_candidates(graph, candidates, seen)
    assert max(seen.values()) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    graph = dataset.build_graph()
    candidates = [State(minute=26, pos=('AA', 'AA'), pressure=0, opened=(), seen=('AA',))]
