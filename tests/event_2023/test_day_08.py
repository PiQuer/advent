import re
from functools import reduce
from itertools import cycle, accumulate, takewhile, islice, chain
from operator import mul
from typing import Iterator

import pytest
from more_itertools import countable, consume
from primefac import primefac

from utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def directions(self) -> Iterator[str]:
        return cycle(self.lines()[0])

    def network(self) -> dict[str, tuple[str, str]]:
        return {match[1]: (match[2], match[3]) for match in
                map(lambda x: re.search(r"(\S{3}) = \((\S{3}), (\S{3})\)", x), self.lines()[2:])}

    def start(self) -> list[str]:
        return [n for n in self.network().keys() if n.endswith("A")]

    def cycle(self, start: str):
        network = self.network()
        yield from accumulate(self.directions(), lambda c, d: network[c][0] if d == "L" else network[c][1],
                              initial=start)

round_1 = dataset_parametrization(year="2023", day="08", examples=[("1", 2), ("2", 6)], result=16343,
                                  dataset_class=DataSet)
round_2 = dataset_parametrization(year="2023", day="08", examples=[("3", 6)], result=15299095336639,
                                  dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    directions = enumerate(dataset.directions())
    network = dataset.network()
    cursor = 'AAA'
    step = 0
    while not cursor == 'ZZZ':
        step, direction = next(directions)
        cursor = network[cursor][0] if direction == "L" else network[cursor][1]
    assert step + 1 == dataset.result

@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    # noinspection PyUnresolvedReferences
    def cycle_length(start):
        node_cycle = countable(islice(dataset.cycle(start), 1, None))
        consume(takewhile(lambda x: not x.endswith('Z'), node_cycle))
        return node_cycle.items_seen

    factors = {len(dataset.lines()[0])}
    factors.update(chain.from_iterable(map(primefac, (map(cycle_length, dataset.start())))))
    assert reduce(mul, factors) == dataset.result
