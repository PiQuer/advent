import pytest
import numpy as np
from collections import defaultdict
from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization("2015", "03", examples=[("1", 2), ("2", 4), ("3", 2)], result=2592)
round_2 = dataset_parametrization("2015", "03", examples=[("4", 3), ("2", 3), ("3", 11)], result=2360)


moves = {
    '^': np.array((0, 1)),
    '>': np.array((1, 0)),
    'v': np.array((0, -1)),
    '<': np.array((-1, 0))
}


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    pos = np.array((0, 0))
    seen = defaultdict(lambda: 0)
    seen[tuple(pos)] = 1
    directions = dataset.text()
    for move in directions:
        pos += moves[move]
        seen[tuple(pos)] += 1
    assert len(seen.keys()) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    pos = [np.array((0, 0)), np.array((0, 0))]
    seen = defaultdict(lambda: 0)
    seen[(0, 0)] += 2
    directions = dataset.text()
    for num, move in enumerate(directions):
        pos[num % 2] += moves[move]
        seen[tuple(pos[num % 2])] += 1
    assert len(seen.keys()) == dataset.result
