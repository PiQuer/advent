import pytest
import numpy as np
from itertools import repeat, pairwise, starmap
from more_itertools import consume

from utils import dataset_parametrization, DataSetBase, np_directions


class DataSet(DataSetBase):
    def directions(self):
        d = np_directions()
        for line in self.lines():
            yield from repeat(d[line[0].lower()], int(line[2:]))


round_1 = dataset_parametrization(day="09", examples=[("1", 13)], result=5513, dataset_class=DataSet)
round_2 = dataset_parametrization(day="09", examples=[("1", 1), ("2", 36)], result=2427, dataset_class=DataSet)


def update_pos(head: np.array, tail: np.array):
    lead = head - tail
    if np.max(np.abs(lead)) > 1:
        tail[:] += np.clip(lead, -1, 1)


class Day09:
    length = 0

    def test_puzzle(self, dataset: DataSet):
        pos = [np.array((0, 0)) for _ in range(self.length)]
        seen = {tuple(pos[-1])}
        for d in dataset.directions():
            pos[0][:] += d
            consume(starmap(update_pos, pairwise(pos)))
            seen.add(tuple(pos[-1]))
        assert len(seen) == dataset.result


@pytest.mark.parametrize(**round_1)
class TestRound1(Day09):
    length = 2


@pytest.mark.parametrize(**round_2)
class TestRound2(Day09):
    length = 10