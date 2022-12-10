import pytest
from itertools import repeat, starmap
from operator import abs, sub, add
try:
    from itertools import pairwise
except ImportError:
    from utils import pairwise
from more_itertools import consume

from utils import dataset_parametrization, DataSetBase, directions


class DataSet(DataSetBase):
    def directions(self):
        d = directions()
        for line in self.lines():
            yield from repeat(d[line[0].lower()], int(line[2:]))


round_1 = dataset_parametrization(day="09", examples=[("1", 13)], result=5513, dataset_class=DataSet)
round_2 = dataset_parametrization(day="09", examples=[("1", 1), ("2", 36)], result=2427, dataset_class=DataSet)


def update_pos(head: list[int], tail: list[int]):
    lead = list(map(sub, head, tail))
    if any(x > 1 for x in map(abs, lead)):
        tail[:] = map(add, tail, map(lambda x: -1 if x < -1 else (1 if x > 1 else x), lead))


class Day09:
    length = 0

    def test_puzzle(self, dataset: DataSet):
        pos = [[0, 0] for _ in range(self.length)]
        seen = {tuple(pos[-1])}
        for d in dataset.directions():
            pos[0][:] = map(add, pos[0], d)
            consume(starmap(update_pos, pairwise(pos)))
            seen.add(tuple(pos[-1]))
        assert len(seen) == dataset.result


@pytest.mark.parametrize(**round_1)
class TestRound1(Day09):
    length = 2


@pytest.mark.parametrize(**round_2)
class TestRound2(Day09):
    length = 10
