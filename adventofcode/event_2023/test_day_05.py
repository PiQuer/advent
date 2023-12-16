"""
--- Day 5: If You Give A Seed A Fertilizer ---
https://adventofcode.com/2023/day/5
"""
import functools
import re
from dataclasses import dataclass
from itertools import chain
from typing import Union, Iterator

from more_itertools import chunked
from sortedcontainers.sortedlist import SortedList

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_rounds


@dataclass
class Range:
    start: int
    length: int

    @property
    def end(self):
        return self.start + self.length

    def intersect(self, other: "Range") -> tuple["Range", "Range"]:
        return Range(other.start, max(min(self.end, other.end)-other.start, 0)), \
            Range(s:=max(self.end, other.start), max(other.end-s, 0))

    def __contains__(self, item: int) -> bool:
        return self.start <= item < self.start + self.length

    def __bool__(self):
        return bool(self.length)

    def __lt__(self, other: "Range") -> bool:
        return self.start < other.start


@dataclass
class Mapping:
    destination: int
    source: Range

    def __getitem__(self, item: Range) -> tuple[Range, Range]:
        mapped, rest = self.source.intersect(item)
        return Range(self.destination+mapped.start-self.source.start, mapped.length), rest

    def __lt__(self, other: Union["Mapping", Range]) -> bool:
        key = other.source.start if isinstance(other, Mapping) else other.start
        return self.source.start < key


@dataclass
class Map:
    map_from: str
    map_to: str
    mappings: SortedList[Mapping]

    def __getitem__(self, item: Range) -> Iterator[Range]:
        index = self.mappings.bisect_right(Mapping(0, item))
        left = self.mappings[index-1] if index>0 else Mapping(item.start, Range(item.start, 0))
        right = self.mappings[index] if index<len(self.mappings) else Mapping(item.end, Range(item.end, 0))
        left_mapped, rest = left[item]
        if left_mapped:
            yield left_mapped
        if rest:
            yield Range(rest.start, min(rest.length, right.source.start - rest.start))
        if right.source.start in rest:
            rest = Range(right.source.start, rest.end - right.source.start)
            yield from self[rest]


def line_to_mapping(line: str) -> Mapping:
    destination, source, length = map(int, line.split())
    return Mapping(destination, Range(source, length))


class DataSet(DataSetBase):
    @functools.cache
    def get_mappings(self) -> tuple[list[int], dict[str, Map]]:
        blocks = self.separated_by_empty_line()
        seeds = list(map(lambda x: int(x[0]), re.finditer(r'\d+', blocks[0])))
        maps = {}
        for block in blocks[1:]:
            names = re.match(r"(\S+)-to-(\S+) map:", block)
            map_from, map_to = names[1], names[2]
            mappings = SortedList(line_to_mapping(line) for line in block.split('\n')[1:])
            maps[map_from] = Map(map_from, map_to, mappings)
        return seeds, maps

    def __hash__(self):
        return hash(self.input_file)


round_1 = dataset_parametrization(year="2023", day="05", examples=[("", 35)], result=88151870, dataset_class=DataSet,
                                  ranges=False)
round_2 = dataset_parametrization(year="2023", day="05", examples=[("", 46)], result=2008785, dataset_class=DataSet,
                                  ranges=True)
pytest_generate_tests = generate_rounds(round_1, round_2)

def test_day_5(dataset: DataSet):
    seeds, mappings = dataset.get_mappings()
    if dataset.params['ranges']:
        seeds_ranges = list(Range(*t) for t in chunked(seeds, 2))
    else:
        seeds_ranges = list(Range(s, 1) for s in seeds)

    def map_to_end(seed: Range):
        category = "seed"
        result = [seed]
        while category in mappings:
            result = list(chain.from_iterable(mappings[category][s] for s in result))
            category = mappings[category].map_to
        return result
    assert min(chain.from_iterable(map(map_to_end, seeds_ranges))).start == dataset.result
