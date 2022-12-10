import pytest
from itertools import repeat
import numpy as np
from collections import defaultdict

from utils import dataset_parametrization, DataSetBase, np_directions


round_1 = dataset_parametrization(day="03", examples=[("1", 6), ("2", 159), ("3", 135)], result=266)
round_2 = dataset_parametrization(day="03", examples=[("1", 30), ("2", 610), ("3", 410)], result=19242)
directions = np_directions()


def extend(wire: set, instruction: str, pos: np.array):
    direction = directions[instruction[0].lower()]
    num = int(instruction[1:])
    wire.update(tuple(pos := pos + d) for d in repeat(direction, num))
    return pos


def extend_delay(wire: defaultdict, instruction: str, pos: np.array, delay: int):
    direction = directions[instruction[0].lower()]
    num = int(instruction[1:])
    for d in repeat(direction, num):
        pos += d
        wire[tuple(pos)] = min(delay := delay+1, wire[tuple(pos)])
    return pos, delay


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    it = iter(dataset.lines())
    wire1 = set()
    wire2 = set()
    for wire in (wire1, wire2):
        pos = np.array((0, 0), dtype=np.int32)
        for inst in next(it).split(','):
            pos = extend(wire, inst, pos)
    assert min(np.sum(np.abs(x)) for x in wire1 & wire2) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    it = iter(dataset.lines())
    wire1 = defaultdict(lambda: np.inf)
    wire2 = defaultdict(lambda: np.inf)
    for wire in (wire1, wire2):
        delay = 0
        pos = np.array((0, 0), dtype=np.int32)
        for inst in next(it).split(','):
            pos, delay = extend_delay(wire, inst, pos, delay)
    assert min(wire1[x] + wire2[x] for x in wire1.keys() & wire2.keys()) == dataset.result
