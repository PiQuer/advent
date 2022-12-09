import pytest
from itertools import repeat
import numpy as np

from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(day="03", examples=[("1", 6), ("2", 159), ("3", 135)], result=266)


def extend(wire: set, instruction: str, pos: np.array):
    direction = np.array({'R': (1, 0), 'L': (-1, 0), 'U': (0, 1), 'D': (0, -1)}[instruction[0]], dtype=np.int32)
    num = int(instruction[1:])
    wire.update(tuple(pos := pos + d) for d in repeat(direction, num))
    return pos


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
