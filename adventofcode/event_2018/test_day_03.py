"""
--- Day 3: No Matter How You Slice It ---
https://adventofcode.com/2018/day/3
"""
import re
from typing import Iterator

import numpy as np
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def claims(self) -> Iterator[tuple[int, int, int, int, int]]:
        for claim in self.lines():
            yield (int(g) for g in re.match(r"#(\d+) @ (\d+),(\d+): (\d+)x(\d+)", claim).groups())


round_1 = dataset_parametrization("2018", "03", examples=[("", 4)], result=118322, dataset_class=DataSet)
round_2 = dataset_parametrization("2018", "03", examples=[("", 3)], result=1178, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    patch = np.zeros((1000, 1000), dtype=np.int32)
    for _, y, x, height, width in dataset.claims():
        patch[y:y+height, x:x+width] += 1
    assert np.sum(patch > 1) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    patch = np.zeros((1000, 1000), dtype=np.int32)
    for _, y, x, height, width in dataset.claims():
        patch[y:y+height, x:x+width] += 1
    assert next(p_id for p_id, y, x, height, width in dataset.claims() if np.all(patch[y:y+height, x:x+width] == 1)) \
        == dataset.result
