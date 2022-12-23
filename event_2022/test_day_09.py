import pytest
import tinyarray as ta
from itertools import repeat, accumulate

from utils import dataset_parametrization, DataSetBase, ta_directions


class DataSet(DataSetBase):
    def directions(self):
        d = ta_directions()
        for line in self.lines():
            yield from repeat(d[line[0].lower()], int(line[2:]))


round_1 = dataset_parametrization(day="09", examples=[("1", 13)], result=5513, dataset_class=DataSet, len=2)
round_2 = dataset_parametrization(day="09", examples=[("1", 1), ("2", 36)], result=2427, dataset_class=DataSet, len=10)


def update_pos(head: ta.array, tail: ta.array):
    lead = head - tail
    if max(ta.abs(lead)) > 1:
        lead = ta.array(tuple(-1 if x < -1 else (1 if x > 1 else x) for x in lead), int)
        return tail + lead
    return tail


# noinspection PyMethodMayBeStatic
class Day09:
    def test_puzzle(self, dataset: DataSet):
        pos = [ta.array((0, 0), int) for _ in range(dataset.params["len"])]
        seen = {pos[-1]}
        for d in dataset.directions():
            pos[0] += d
            pos = list(accumulate(pos, update_pos))
            seen.add(pos[-1])
        assert len(seen) == dataset.result


@pytest.mark.parametrize(**round_1)
class TestRound1(Day09):
    pass


@pytest.mark.parametrize(**round_2)
class TestRound2(Day09):
    pass
