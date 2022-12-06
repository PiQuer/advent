import pytest
from itertools import combinations
import numpy as np
from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization("02", examples=[("1", 12)], result=5658)
round_2 = dataset_parametrization("02", examples=[("2", "fgij")], result="nmgyjkpruszlbaqwficavxneo")


@pytest.mark.parametrize(**round_1)
def test_round1(dataset: DataSetBase):
    result = np.array((0, 0))
    for line in dataset.lines():
        counts = [line.count(x) for x in set(line)]
        result += np.array((2 in counts, 3 in counts))
    assert np.prod(result) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round2(dataset: DataSetBase):
    candidate = next((a, b) for a, b in combinations(dataset.lines(), 2)
                     if sum(1 for c1, c2 in zip(a, b) if c1 != c2) == 1)
    assert ''.join(c1 for c1, c2 in zip(*candidate) if c1 == c2) == dataset.result
