import heapq
from dataclasses import dataclass
from functools import cached_property, cache
from itertools import count

import numpy as np
import pytest
import tinyarray as ta

from utils import dataset_parametrization, DataSetBase

# from utils import generate_rounds

year="2023"
day="14"

@dataclass(frozen=True)
class Rock:
    coordinates: ta.ndarray_int

class RollingNorth(Rock):
    def __lt__(self, other: Rock):
        return self.coordinates[0] < other.coordinates[0] or \
            self.coordinates[0] == other.coordinates[0] and self.coordinates[1] < other.coordinates[1]

class RollingWest(Rock):
    def __lt__(self, other: Rock):
        return self.coordinates[1] < other.coordinates[1] or \
            self.coordinates[1] == other.coordinates[1] and self.coordinates[0] < other.coordinates[0]

class RollingSouth(Rock):
    def __lt__(self, other: Rock):
        return self.coordinates[0] > other.coordinates[0] or \
            self.coordinates[0] == other.coordinates[0] and self.coordinates[1] > other.coordinates[1]

class RollingEast(Rock):
    def __lt__(self, other: Rock):
        return self.coordinates[1] > other.coordinates[1] or \
            self.coordinates[1] == other.coordinates[1] and self.coordinates[0] > other.coordinates[0]

AnyRock = RollingNorth | RollingWest | RollingSouth | RollingEast
direction_classes: dict[int, type[AnyRock]] = {0: RollingNorth, 1: RollingWest, 2: RollingSouth, 3: RollingEast}
direction_keys = {v: k for k, v in direction_classes.items()}

class DataSet(DataSetBase):
    @cached_property
    def array(self) -> np.ndarray:
        return self.np_array_bytes()

    def cube_rocks(self) -> dict[ta.ndarray_int, bool]:
        return {ta.array(a): True for a in np.argwhere(self.array == b'#').reshape((-1, 2))}

    def rocks(self) -> list[RollingNorth]:
        yield from map(RollingNorth, map(ta.array, np.argwhere(self.array == b'O').reshape((-1, 2))))

round_1 = dataset_parametrization(year=year, day=day, examples=[("", 136)], result=109385, dataset_class=DataSet)
round_2 = dataset_parametrization(year=year, day=day, examples=[("", 64)], result=93102, dataset_class=DataSet)
# pytest_generate_tests = generate_rounds(round_1, round_2)

@cache
def abort_condition(rock: AnyRock, direction_key: int, shape: tuple[int, ...]) -> bool:
    return (lambda c: c[0] == 0, lambda c: c[1] == 0, lambda c: c[0] == shape[0] - 1, lambda c: c[1] == shape[1] -1) \
        [direction_key](rock.coordinates)

@cache
def get_next_rock(rock: AnyRock, direction_key: int) -> AnyRock:
    next_class = type(rock)
    return (lambda c: next_class(c-(1, 0)),
            lambda c: next_class(c-(0, 1)),
            lambda c: next_class(c+(1, 0)),
            lambda c: next_class(c+(0, 1)))[direction_key](rock.coordinates)

def roll(cube_rocks: dict[ta.ndarray_int, bool], rocks: list[AnyRock], shape: tuple[int, ...]) -> list[AnyRock]:
    result = []
    rock_map = {}
    direction_key = direction_keys[type(rocks[0])]
    next_class = direction_classes[(direction_key + 1) % 4]
    while rocks:
        this_rock = rocks.pop()
        # noinspection PyArgumentList
        rock = next_class(this_rock.coordinates)
        next_rock = get_next_rock(rock, direction_key)
        while not abort_condition(rock, direction_key, shape) and not rock_map.get(next_rock) \
                and not cube_rocks.get(next_rock.coordinates):
            rock = next_rock
            next_rock = get_next_rock(next_rock, direction_key)
        rock_map[rock] = True
        result.append(rock)
    return sorted(result, reverse=True)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    cube_rocks = dataset.cube_rocks()
    rocks = list(dataset.rocks())
    heapq.heapify(rocks)
    rocks = roll(cube_rocks, rocks, dataset.array.shape)
    assert sum(dataset.array.shape[0] - r.coordinates[0] for r in rocks) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    cube_rocks = dataset.cube_rocks()
    rocks = sorted(dataset.rocks(), reverse=True)
    seen = {tuple(rocks): 0}
    cycle = [tuple(rocks)]
    index = 0
    for index in count(1):
        for _ in range(4):
            rocks = roll(cube_rocks, rocks, dataset.array.shape)
        if (h := tuple(rocks)) in seen:
            break
        seen[h] = index
        cycle.append(h)
    final = cycle[(1000000000 - index) % (index - seen[h]) + seen[h]]
    assert sum(dataset.array.shape[0] - r.coordinates[0] for r in final) == dataset.result
