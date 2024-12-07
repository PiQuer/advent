"""
--- Day 5: Print Queue ---
https://adventofcode.com/2024/day/05
"""
from typing import Iterator

import networkx as nx

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_parts

YEAR= "2024"
DAY= "05"

class DataSet(DataSetBase):
    def graph(self) -> nx.DiGraph:
        result = nx.DiGraph()
        rules, _ = self.separated_by_empty_line()
        for line in rules.splitlines():
            result.add_edge(*map(int, line.split('|', maxsplit=1)))
        return result

    def updates(self) -> Iterator[dict[int, int]]:
        update_lines = self.separated_by_empty_line()[1].splitlines()
        return map(lambda x: dict((int(v), k) for k, v in enumerate(x.split(','))), update_lines)


part_1 = dataset_parametrization(year=YEAR, day=DAY, part=1, dataset_class=DataSet)
part_2 = dataset_parametrization(year=YEAR, day=DAY, part=2, dataset_class=DataSet)

pytest_generate_tests = generate_parts(part_1, part_2)

def sort(graph: nx.DiGraph, update: dict[int, int]) -> list[int]:
    return list(nx.lexicographical_topological_sort(graph.subgraph(update.keys()), key=update.get))


def test_both_parts(dataset: DataSet):
    graph = dataset.graph()
    def middle_number(update: dict[int, int]) -> int:
        sorted = sort(graph, update)
        middle_index = (len(sorted)-1)//2
        match dataset.part:
            case 1:
                return sorted[middle_index] if sorted == list(update.keys()) else 0
            case 2:
                return sorted[middle_index] if sorted != list(update.keys()) else 0
    dataset.assert_answer(sum(map(middle_number, dataset.updates())))
