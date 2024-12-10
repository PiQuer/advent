"""
--- Day 7: Bridge Repair ---
https://adventofcode.com/2024/day/07
"""
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Callable

from adventofcode.utils import dataset_parametrization, DataSetBase
from adventofcode.utils import generate_parts

YEAR= "2024"
DAY= "07"


Operator = Callable[int, [int, int]]


@dataclass(frozen=True)
class Equation:
    test_value: int
    numbers: list[int]


class DataSet(DataSetBase):
    def preprocess(self) -> Iterator[tuple[int, list[int]]]:
        def process_line(line: str):
            test_value, list_of_numbers = line.split(': ', maxsplit=1)
            return Equation(int(test_value), list(map(int, list_of_numbers.split())))
        yield from map(process_line, self.lines())


part_1 = dataset_parametrization(year=YEAR, day=DAY, part=1, dataset_class=DataSet)
part_2 = dataset_parametrization(year=YEAR, day=DAY, part=2, dataset_class=DataSet, example_results=[11387])
pytest_generate_tests = generate_parts(part_1, part_2)


def branch(equation: Equation, with_concat=False) -> Iterator[Equation]:
    numbers = equation.numbers[:-1]
    number = equation.numbers[-1]
    test_values = []
    if equation.test_value > number:
        test_values.append(equation.test_value - number)
        if with_concat:
            if repr(equation.test_value).endswith(repr(number)):
                test_values.append(int(repr(equation.test_value)[:-len(repr(number))]))
    if equation.test_value % number == 0:
        test_values.append(equation.test_value // number)
    for test_value in test_values:
        if test_value >= numbers[-1]:
            yield Equation(test_value, numbers)


def find_solution(backlog: deque[Equation], with_concat=False) -> bool:
    equation = backlog.pop()
    if len(equation.numbers) == 1:
        if equation.test_value == equation.numbers[0]:
            return True
        return False
    backlog.extend(branch(equation, with_concat=with_concat))


def test_both_parts(dataset: DataSet):
    def solve(equation: Equation) -> int:
        backlog = deque((equation,))
        results = []
        while backlog:
            if find_solution(backlog, with_concat=(dataset.part == 2)):
                return equation.test_value
        return 0
    dataset.assert_answer(sum(map(solve, dataset.preprocess())))
