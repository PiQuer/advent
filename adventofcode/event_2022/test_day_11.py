"""
--- Day 11: Monkey in the Middle ---
https://adventofcode.com/2022/day/11
"""
import re
from collections import deque
from operator import pow as op_pow
from operator import add, mul
from typing import Iterable, Callable, Any

from more_itertools import partition

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_parts


class Monkey:
    #pylint: disable=too-many-arguments
    def __init__(self, worry_levels: Iterable[Any], op: Callable[[int, int], int], const: int, divider: int,
                 target_true: int, target_false: int):
        self._op = op
        self._const = const
        self._wl = deque(worry_levels)
        self._divider = divider
        self._target_true = target_true
        self._target_false = target_false
        self._activity = 0

    @property
    def activity(self):
        return self._activity

    def _inspect(self, wl: Any):
        return self._op(wl, self._const) // 3

    def _test(self, wl: Any):
        return wl % self._divider == 0

    def throw(self, targets: dict[int, "Monkey"]):
        it_false, it_true = partition(self._test, map(self._inspect, self._wl))
        targets[self._target_true].catch(it_true)
        targets[self._target_false].catch(it_false)
        self._activity += len(self._wl)
        self._wl.clear()

    def catch(self, items: Iterable[int]):
        self._wl.extend(items)

    def __repr__(self):
        return f"Monkey(worrey_levels={self._wl}, op={self._op.__name__}, const={self._const}, " \
               "divider={self._divider}, target_true={self._target_true}, target_false={self._target_false})"


class Monkey2(Monkey):
    def _inspect(self, wl: dict[int, int]):
        return {key: self._op(value, self._const) % key for key, value in wl.items()}

    def _test(self, wl: dict[int, int]):
        return wl[self._divider] == 0


# noinspection PyMethodMayBeStatic
class DataSet(DataSetBase):
    def _wl(self, wl_str):
        return (int(x) for x in wl_str.split(', '))

    def _monkey_factory(self, *args, **kwargs):
        return Monkey(*args, **kwargs)

    def monkeys(self):
        result = {}
        for m in self.separated_by_empty_line():
            match = re.search(r"Monkey (\d+):.*items: ([\d, ]+).*new = (.*)\n.*by (\d+).*monkey (\d+).*monkey (\d+)", m,
                              re.MULTILINE | re.DOTALL)
            op_match = match.group(3)[4:]
            if op_match == "* old":
                op, const = op_pow, 2
            elif "+" in op_match:
                op, const = add, int(op_match[2:])
            elif "*" in op_match:
                op, const = mul, int(op_match[2:])
            else:
                raise NotImplementedError
            result[int(match.group(1))] = \
                self._monkey_factory(worry_levels=self._wl(match.group(2)), op=op, const=const,
                                     divider=int(match.group(4)), target_true=int(match.group(5)),
                                     target_false=int(match.group(6)))
        return result


class DataSet2(DataSet):
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23)

    def _monkey_factory(self, *args, **kwargs):
        return Monkey2(*args, **kwargs)

    def _wl(self, wl_str):
        return ({p: wl % p for p in self.primes} for wl in (int(x) for x in wl_str.split(', ')))


round_1 = dataset_parametrization(year="2022", day="11",
                                  examples=[("", 10605)], result=55216, dataset_class=DataSet, rounds=20)
round_2 = dataset_parametrization(year="2022", day="11",
                                  examples=[("", 2713310158)], result=12848882750, dataset_class=DataSet2,
                                  rounds=10000)
pytest_generate_tests = generate_parts(round_1, round_2)


def test_day_11(dataset: DataSet):
    monkeys = dataset.monkeys()
    for _ in range(dataset.params["rounds"]):
        for m_id in monkeys:
            monkeys[m_id].throw(monkeys)
    assert mul(*sorted(m.activity for m in monkeys.values())[-2:]) == dataset.result
