"""
--- Day 4: Scratchcards ---
https://adventofcode.com/2023/day/4
"""
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase


@dataclass
class Card:
    index: int
    winning: set[str]
    own: set[str]


class DataSet(DataSetBase):
    def preprocessed(self) -> Iterator[Card]:
        for line in self.lines():
            match = re.match(r"Card +(\d+): +([\d ]*) \| +([\d ]*)", line)
            yield Card(int(match[1]), *(map(lambda x: set(x.split()), (match[2], match[3]))))

round_1 = dataset_parametrization(year="2023", day="04", examples=[("", 13)], part=1, dataset_class=DataSet)
round_2 = dataset_parametrization(year="2023", day="04", examples=[("", 30)], part=2, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    result = sum((2**(l-1) if (l:=len(card.own&card.winning)) else 0)  for card in dataset.preprocessed())
    dataset.assert_answer(result)


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    max_index = len(dataset.lines())
    copies = defaultdict(lambda: 1)
    for card in dataset.preprocessed():
        for idx in range(card.index + 1, min(max_index + 1, card.index + 1 + len(card.own&card.winning))):
            copies[idx] += copies[card.index]
    dataset.assert_answer(sum(copies[idx] for idx in range(1, max_index + 1)))
