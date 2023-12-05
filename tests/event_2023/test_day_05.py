from dataclasses import dataclass
import functools
import re

import pytest
from sortedcontainers.sortedlist import SortedList

from utils import dataset_parametrization, DataSetBase


@dataclass
class Mapping:
    destination: int
    source: int
    length: int

    def __contains__(self, item: int) -> bool:
        return self.source <= item < self.source + self.length

    def __getitem__(self, item: int) -> int:
        return self.destination + (item - self.source) if item in self else item

    def __lt__(self, other: "Mapping") -> bool:
        key = other if isinstance(other, int) else other.source
        return self.source < key


@dataclass
class Map:
    map_from: str
    map_to: str
    mappings: SortedList[Mapping]

    def __getitem__(self, item: int) -> int:
        index = self.mappings.bisect_right(Mapping(0, item, 0))
        return item if index == 0 else self.mappings[index-1][item]


class DataSet(DataSetBase):
    @functools.cache
    def get_mappings(self) -> tuple[list[int], dict[str, Map]]:
        blocks = self.separated_by_empty_line()
        seeds = list(map(lambda x: int(x[0]), re.finditer(r'\d+', blocks[0])))
        maps = {}
        for block in blocks[1:]:
            names = re.match(r"(\S+)-to-(\S+) map:", block)
            map_from, map_to = names[1], names[2]
            mappings = SortedList(Mapping(*map(int, line.split())) for line in block.split('\n')[1:])
            maps[map_from] = Map(map_from, map_to, mappings)
        return seeds, maps

    def __hash__(self):
        return hash(self.input_file)


round_1 = dataset_parametrization(year="2023", day="05", examples=[("", 35)], result=88151870, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    seeds, mappings = dataset.get_mappings()

    def map_to_end(seed: int):
        category = "seed"
        result = seed
        while category in mappings:
            result = mappings[category][result]
            category = mappings[category].map_to
        return result
    assert min(map(map_to_end, seeds)) == dataset.result
