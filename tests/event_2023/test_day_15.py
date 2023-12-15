"""
--- Day 15: Lens Library ---
https://adventofcode.com/2023/day/15
"""
from dataclasses import dataclass, field
from functools import reduce, cache

import pytest

from utils import dataset_parametrization, DataSetBase

year="2023"
day="15"

round_1 = dataset_parametrization(year=year, day=day, examples=[("", 1320)], result=507291, dataset_class=DataSetBase)
round_2 = dataset_parametrization(year=year, day=day, examples=[("", 145)], result=296921, dataset_class=DataSetBase)


def update(current: int, char: int) -> int:
    return ((current + char) * 17) % 256


@cache
def calculate_hash(b: bytes) -> int:
    return reduce(update, bytearray(b), 0)


@dataclass
class Lens:
    label: bytes
    focal_length: int | bytes

    def __post_init__(self):
        self.focal_length = int(self.focal_length)


@dataclass
class Box:
    label: int
    lenses: dict[bytes, Lens] = field(default_factory=dict)

    def replace(self, lens: Lens):
        self.lenses[lens.label] = lens

    def remove(self, label: bytes):
        if label in self.lenses:
            del self.lenses[label]


class Boxes(dict):
    def __missing__(self, key):
        res = self[key] = Box(key)
        return res


def focusing_power(box: Box) -> int:
    return sum((box.label+1)*(slot+1)*b.focal_length for slot, b in enumerate(box.lenses.values()))


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    assert sum(map(calculate_hash, dataset.bytes().split(b','))) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    boxes = Boxes()
    for lens_bytes in dataset.bytes().split(b','):
        if lens_bytes.endswith(b'-'):
            label = lens_bytes[:-1]
            boxes[calculate_hash(label)].remove(label)
        else:
            lens = Lens(*lens_bytes.split(b'=', maxsplit=2))
            boxes[calculate_hash(lens.label)].replace(lens)
    assert sum(map(focusing_power, boxes.values())) == dataset.result