"""
https://adventofcode.com/2022/day/16
TODO: too slow
"""
import pytest
import networkx as nx
import re
from typing import Iterable, Union, Optional
from dataclasses import dataclass
from itertools import product, chain
from utils import dataset_parametrization, DataSetBase
import logging


@dataclass
class State:
    minute: int
    pressure: int
    pos: str
    opened: Union[tuple[str], tuple[()]]

    def __hash__(self):
        return hash((self.pos, self.opened))

    def __eq__(self, other: "State"):
        return self.pos == other.pos and self.opened == other.opened
    
    
@dataclass
class State2:
    minute: int
    pressure: int
    pos: tuple[str, str]
    opened: dict[int, Union[tuple[str], tuple[()]]]
    previous: Optional["State2"] = None

    def __str__(self):
        return f"minute={self.minute}, pressure={self.pressure}, pos={self.pos}, opened0: {self.opened[0]}" \
            f" opened1: {self.opened[1]}"


class DataSet(DataSetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valves = set()
        self.max = 0
        self.max_state: Optional[State2] = None

    def build_graph(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for line in self.lines():
            match = re.match(r"Valve ([A-Z]{2}) has flow rate=(\d+);.*valves? ([A-Z, ]*)", line)
            node, flow_rate = match.groups()[:2]
            targets = match.group(3).split(', ')
            graph.add_edges_from((node, target, {"flow_rate": 0}) for target in targets)
            if flow_rate:
                graph.add_edge(node, node, flow_rate=int(flow_rate))
                self.valves.add(node)
        return graph


round_1 = dataset_parametrization(day="16", examples=[("", 1651)], result=1559, dataset_class=DataSet)
round_2 = dataset_parametrization(day="16", examples=[("", 1707)], result=2191, dataset_class=DataSet)


def get_next_candidates(graph: nx.MultiDiGraph, candidates: Iterable[State],
                        seen: dict[State, int], dataset: DataSet) -> Iterable[State]:
    result = []
    for c in candidates:
        if len(c.opened) == len(dataset.valves) or not c.minute:
            continue
        next_minute = c.minute - 1
        for nbr in graph[c.pos]:
            nbr: str
            edge = graph[c.pos][nbr][0]
            if (fr := edge['flow_rate']) and nbr not in c.opened:
                candidate = State(minute=next_minute, pressure=c.pressure + fr * next_minute, pos=nbr,
                                  opened=tuple(sorted(c.opened + (nbr,))))
            elif nbr == c.pos:
                continue
            else:
                candidate = State(minute=next_minute, pressure=c.pressure, pos=nbr, opened=c.opened)
            if seen.get(candidate, -1) < candidate.pressure:
                result.append(candidate)
                seen[candidate] = candidate.pressure
    return result


def get_next_candidates2(graph: nx.MultiDiGraph, candidates: Iterable[State2],
                         globally_seen, personally_seen, dataset: DataSet) -> Iterable[State2]:
    result = []
    for c in candidates:
        if len(c.opened) == len(dataset.valves) or not c.minute:
            continue
        next_minute = c.minute - 1
        for n1, n2 in product(graph[c.pos[0]], graph[c.pos[1]]):
            _opened, _pressure = c.opened.copy(), c.pressure
            e1, e2 = graph[c.pos[0]][n1][0], graph[c.pos[1]][n2][0]
            if n1 == n2 and e1['flow_rate'] and e2['flow_rate']:
                continue
            for player, (nbr, e, pos) in enumerate(zip((n1, n2), (e1, e2), c.pos)):
                if (fr := e['flow_rate']) and nbr not in chain(_opened[0], _opened[1]):
                    _opened[player] = tuple(sorted(_opened[player] + (nbr,)))
                    _pressure += fr * next_minute
            candidate = State2(minute=next_minute, pressure=_pressure, pos=(n1, n2), opened=_opened, previous=c)
            globally_seen_key = (tuple(sorted((n1, n2))), tuple(sorted(chain.from_iterable(_opened.values()))))
            personally_seen_key = [(n, _opened[n[0]]) for n in zip((0, 1), (n1, n2))]
            if globally_seen.get(globally_seen_key, -1) \
                    < candidate.pressure and \
                    personally_seen.get(personally_seen_key[0], -1) < candidate.pressure and \
                    personally_seen.get(personally_seen_key[1], -1) < candidate.pressure:
                if candidate.pressure > dataset.max:
                    dataset.max = candidate.pressure
                    dataset.max_state = candidate
                if (dataset.max - candidate.pressure) <= 0.4 * dataset.max:
                    # Todo: fix this
                    result.append(candidate)
                globally_seen[globally_seen_key] = candidate.pressure
    for c in result:
        personally_seen[((0, c.pos[0]), c.opened[0])] = c.pressure
        personally_seen[((1, c.pos[1]), c.opened[1])] = c.pressure
    return result


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    graph = dataset.build_graph()
    candidates = [State(minute=30, pos='AA', pressure=0, opened=())]
    seen = {candidates[0]: 0}
    while candidates:
        candidates = get_next_candidates(graph, candidates, seen, dataset)
    assert dataset.max == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    graph = dataset.build_graph()
    candidates = [State2(minute=26, pos=('AA', 'AA'), pressure=0, opened={0: (), 1: ()})]
    globally_seen = {(('AA', 'AA'), ()): 0}
    personally_seen = {}
    while candidates:
        logging.info(f"{candidates[0].minute}: {len(candidates)}")
        candidates = get_next_candidates2(graph, candidates, globally_seen, personally_seen, dataset)
    assert dataset.max == dataset.result
    state = dataset.max_state
    while state:
        logging.info(state)
        state = state.previous
