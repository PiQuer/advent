"""
--- Day 12: Hot Springs ---
https://adventofcode.com/2023/day/12
"""
import multiprocessing
import re
from dataclasses import dataclass
from functools import cache
from typing import Iterator

import pytest

from utils import dataset_parametrization, DataSetBase

year="2023"
day="12"


@dataclass
class State:
    condition: str
    contiguous: tuple[int, ...]

    def __hash__(self):
        return hash((self.condition, self.contiguous))


def reduce(condition: str, contiguous: tuple[int, ...]) -> State|None:
    if contiguous == ():
        return None if '#' in condition else State("", ())
    condition = condition.lstrip('.').rstrip('.')
    if len(condition) < sum(contiguous) + len(contiguous) - 1:
        return None
    match = re.search(r'(.*?)(#*)$', condition)
    left, right = match.group(1), match.group(2)
    if (l:=len(right)) > contiguous[-1]:
        return None
    if l == contiguous[-1]:
        return reduce(left[:-1], contiguous[:-1])
    if l > 0:
        if left.endswith('?'):
            return reduce(f"{left[:-1]}#", contiguous[:-1] + (contiguous[-1] - l,))
        return None
    return State(left, contiguous)


@cache
def configurations(state: State|None) -> int:
    if state is None:
        return 0
    if state == State("", ()):
        return 1
    assert state.condition.endswith('?')
    a, b = reduce(state.condition[:-1], state.contiguous), \
        reduce(f"{state.condition[:-1]}#", state.contiguous)
    return configurations(a) + configurations(b)


class DataSet(DataSetBase):
    def states(self) -> Iterator[State]:
        yield from (reduce(con, tuple(map(int, cont.split(','))))
                    for con, cont in map(lambda l: l.split(), self.lines()))

    def folded_states(self) -> Iterator[State]:
        yield from (reduce('?'.join((con,)*5), tuple(map(int, cont.split(',')))*5)
                    for con, cont in map(lambda l: l.split(), self.lines()))

round_1 = dataset_parametrization(year=year, day=day, examples=[("", 21)], result=7195, dataset_class=DataSet)
round_2 = dataset_parametrization(year=year, day=day, examples=[("", 525152)], result=33992866292225,
                                  dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    assert sum(map(configurations, dataset.states())) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    pool = multiprocessing.Pool()
    assert sum(pool.map(configurations, dataset.folded_states())) == dataset.result


@pytest.mark.parametrize("con,cont,answer",
                         (("...#?###.", (1, 2), None),
                          ("...?##", (2,), State("", ())),
                          ("...?.##...", (1, 2), State("?", (1,))),
                          ("###?###", (1, 4), None),
                          ("..#.####", (1, 4), State("", ())),
                          ("##.##", (4,), None)))
def test_reduce(con, cont, answer):
    assert reduce(con, cont) == answer


@pytest.mark.parametrize("con,cont,answer",
                         (("???.###", (1,1,3), 1),
                          (".??..??...?##.", (1,1,3), 4),
                          ("?#?#?#?#?#?#?#?", (1,3,1,6), 1),
                          ("????.#...#...", (4,1,1), 1),
                          ("????.######..#####.", (1,6,5), 4),
                          ("?###????????", (3,2,1), 10)))
def test_configurations(con, cont, answer):
    assert configurations(reduce(con, cont)) == answer