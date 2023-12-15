"""
--- Day 4: Security Through Obscurity ---
https://adventofcode.com/2016/day/4
"""
import re
from collections import Counter
from itertools import islice

import pytest

from utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def preprocess(self):
        for line in self.lines():
            m = re.match(r"([a-z-]+)-(\d+)\[([a-z]+)]", line)
            count, sector_id, checksum = Counter(m.group(1).replace('-', '')), int(m.group(2)), m.group(3)
            if ''.join(islice((a for a, _ in sorted(count.items(), key=lambda y: (-y[1], y[0]))), 5)) \
                    == checksum:
                yield sector_id, m.group(1)


round_1 = dataset_parametrization("2016", "04", examples=[("", 1514)], result=361724, dataset_class=DataSet)
round_2 = dataset_parametrization("2016", "04", examples=[], result=482, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    assert sum(v for v, _ in dataset.preprocess()) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert next(sector_id for sector_id, name in dataset.preprocess() if
                ''.join(chr((ord(i) + sector_id - 97) % 26 + 97) if i != '-' else ' ' for i in name) ==
                "northpole object storage") == dataset.result
