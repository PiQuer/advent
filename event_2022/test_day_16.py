"""
https://adventofcode.com/2022/day/16
"""
from functools import cache
import networkx as nx
import re
from typing import Optional
from dataclasses import dataclass, field
from itertools import product, combinations
from more_itertools import padded, value_chain
import heapq

from utils import dataset_parametrization, DataSetBase, generate_rounds


@dataclass
class State:
    minute: int
    pressure: int
    pos: tuple[str, ...]
    opened: dict[int, frozenset[str]]
    all_opened: frozenset[str]
    previous: Optional["State"] = None

    def __str__(self):
        return f"minute={self.minute}, pressure={self.pressure}, pos={self.pos}, opened: {self.opened}"

    def __lt__(self, other: "State"):
        return self.pressure > other.pressure


class DataSet(DataSetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valves = frozenset()
        self.max = 0
        self.max_state: Optional[State] = None

    def build_graph(self) -> nx.Graph:
        graph = nx.Graph()
        valves = set()
        for line in self.lines():
            match = re.match(r"Valve ([A-Z]{2}) has flow rate=(\d+);.*valves? ([A-Z, ]*)", line)
            node, flow_rate = match.groups()[:2]
            flow_rate = int(flow_rate)
            targets = match.group(3).split(', ')
            graph.add_node(node, flow_rate=flow_rate)
            graph.add_edges_from((node, target) for target in targets)
            if int(flow_rate):
                valves.add(node)
        self.valves = frozenset(valves)
        return graph


@dataclass
class Seen:
    glob: dict[tuple, tuple[int, int]] = field(default_factory=dict)
    ind: dict[int, dict[tuple[str, frozenset[str]], int]] = field(default_factory=lambda: {0: dict(), 1: dict()})


class DeadEnd(Exception):
    pass


def get_next_candidates(candidates: list[State], graph: nx.Graph, seen: Seen, dataset: DataSet, players: int):
    c = heapq.heappop(candidates)
    next_minute = c.minute - 1
    if next_minute == 0:
        return
    for p in product(*(value_chain(c.pos[p_id], graph[c.pos[p_id]]) for p_id in range(players))):
        all_opened, pressure, opened = c.all_opened, c.pressure, c.opened.copy()
        try:
            all_opened, pressure = open_valve(all_opened, c, graph, next_minute, opened, p, pressure, seen)
        except DeadEnd:
            continue
        if (glob_key := tuple(sorted(p)) + (all_opened,)) in seen.glob:
            if seen.glob[glob_key][0] >= next_minute and seen.glob[glob_key][1] >= pressure:
                continue
        else:
            seen.glob[glob_key] = (next_minute, pressure)
        for idx, n in enumerate(p):
            seen.ind[idx][(n, opened[idx])] = next_minute
        if pressure > dataset.max:
            dataset.max = pressure
        elif pressure + \
                upper_bound(graph, next_minute, dataset.valves, all_opened, players) <= dataset.max:
            continue
        heapq.heappush(candidates, State(minute=next_minute, pos=tuple(p), pressure=pressure, opened=opened,
                                         all_opened=all_opened, previous=c))


def open_valve(all_opened, c, graph, next_minute, opened, p, pressure, seen):
    for idx, n in enumerate(p):
        if n == c.pos[idx]:
            if graph.nodes[n]["flow_rate"] and n not in all_opened:
                pressure += next_minute * graph.nodes[n]["flow_rate"]
                all_opened |= {n}
                opened[idx] |= {n}
            else:
                raise DeadEnd()
        if (n, opened[idx]) in seen.ind[idx] and seen.ind[idx][(n, opened[idx])] > next_minute:
            raise DeadEnd()
    return all_opened, pressure


@cache
def all_distances(graph: nx.Graph) -> dict:
    return dict(nx.all_pairs_shortest_path_length(graph))


@cache
def remaining_valve_distances(graph: nx.Graph, valves: frozenset[str], opened_valves: frozenset[str]):
    d = all_distances(graph)
    return sorted(d[x[0]][x[1]] for x in combinations(valves - opened_valves, 2))


@cache
def upper_bound(graph: nx.Graph, minutes_remaining: int, valves: frozenset[str], opened_valves: frozenset[str],
                players: int) -> int:
    if minutes_remaining <= 1:
        return 0
    minutes_remaining -= 1
    result = 0
    remaining_valves = padded(
        sorted((graph.nodes[x]["flow_rate"] for x in valves.difference(opened_valves)), reverse=True), fillvalue=0)
    distances = iter(remaining_valve_distances(graph, valves, opened_valves))
    result += minutes_remaining * (r := next(remaining_valves))
    if players == 2:
        result += minutes_remaining * (r := next(remaining_valves))
    try:
        while minutes_remaining and r:
            m1 = next(distances)
            minutes_remaining = max(minutes_remaining - m1 - 1, 0)
            result += minutes_remaining * next(remaining_valves)
            for _ in range(players - 1):
                m2 = next(distances)
                minutes_remaining = max(minutes_remaining - (m2 - m1), 0)
                result += minutes_remaining * (r := next(remaining_valves))
                m1 = m2
    except StopIteration:
        pass
    return result


round_1 = dataset_parametrization(day="16", examples=[("", 1651)], result=1559, dataset_class=DataSet,
                                  players=1, minutes=30)
round_2 = dataset_parametrization(day="16", examples=[("", 1707)], result=2191, dataset_class=DataSet,
                                  players=2, minutes=26)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_puzzle(dataset: DataSet):
    graph = dataset.build_graph()
    minutes, players = dataset.params["minutes"], dataset.params["players"]
    candidates = [State(minute=minutes, pos=('AA',) * players, pressure=0,
                        opened={idx: frozenset() for idx in range(players)}, all_opened=frozenset())]
    seen = Seen()
    seen.glob[('AA', frozenset())] = (minutes, 0)
    for idx in range(players):
        seen.ind[idx][('AA',) * players + (frozenset(),)] = minutes
    while candidates:
        get_next_candidates(candidates, graph, seen, dataset, players)
    assert dataset.max == dataset.result
