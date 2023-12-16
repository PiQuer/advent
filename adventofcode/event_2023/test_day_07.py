"""
--- Day 7: Camel Cards ---
https://adventofcode.com/2023/day/7
"""
from collections import Counter
from dataclasses import dataclass
from functools import cached_property

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_rounds


@dataclass
class Hand:
    cards: str
    bid: int

    def __lt__(self, other: "Hand") -> bool:
        return self.type < other.type or (self.type == other.type and self.lex < other.lex)

    @cached_property
    def lex(self):
        return self.cards.replace("A", "E").replace("K", "D").replace("Q", "C") \
            .replace("T", "A")

    @cached_property
    def type(self):
        counts = Counter(self.lex)
        jokers = counts["1"]
        del counts["1"]
        c = sorted(list(counts.values()))
        result = 0
        if jokers == 5 or c[-1] + jokers == 5:
            result = 6
        elif c[-1] + jokers == 4:
            result = 5
        elif c[-1] + c[-2] + jokers == 5:
            result = 4
        elif c[-1] + jokers == 3:
            result = 3
        elif c[-1] + c[-2] + jokers == 4:
            result = 2
        elif c[-1] + jokers == 2:
            result = 1
        return result


class Hand1(Hand):
    @cached_property
    def lex(self):
        return super().lex.replace("J", "B")


class Hand2(Hand):
    @cached_property
    def lex(self):
        return super().lex.replace("J", "1")


class DataSet(DataSetBase):
    def preprocessed(self, hand_class) -> list[Hand]:
        return [hand_class(cards, int(bid)) for cards, bid in map(lambda l: l.split(), self.lines())]


round_1 = dataset_parametrization(year="2023", day="07", examples=[("", 6440)], result=250602641,
                                  dataset_class=DataSet, hand_class=Hand1)
round_2 = dataset_parametrization(year="2023", day="07", examples=[("", 5905)], result=251037509,
                                  dataset_class=DataSet, hand_class=Hand2)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_day_7(dataset: DataSet):
    assert sum(h.bid * (rank + 1) for rank, h in
               enumerate(sorted(dataset.preprocessed(hand_class=dataset.params["hand_class"])))) == dataset.result
