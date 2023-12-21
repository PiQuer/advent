"""
--- Day 19: Aplenty ---
https://adventofcode.com/2023/day/19
"""
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

# from adventofcode.utils import generate_rounds

YEAR= "2023"
DAY= "19"

@dataclass
class Part:
    x: int
    m: int
    a: int
    s: int

    @property
    def rating(self):
        return self.x + self.m + self.a + self.s

@dataclass
class Rule:
    if_true: str
    field: str | None = None
    operator: Callable[[int, int], bool] | None = None
    value: int | None = None

    def apply(self, part: Part) -> str | None:
        if self.operator is None or self.operator(getattr(part, self.field), self.value):
            return self.if_true
        return None

@dataclass
class Workflow:
    name: str
    rules: list[Rule]

    def rating(self, part: Part, workflows: dict[str, "Workflow"]) -> int:
        for rule in self.rules:
            if (result := rule.apply(part)) is not None:
                match result:
                    case "A":
                        return part.rating
                    case "R":
                        return 0
                    case next_workflow:
                        return workflows[next_workflow].rating(part, workflows)
        assert False

class DataSet(DataSetBase):
    def parse_rule(self, instruction: str) -> Rule:
        if ':' in instruction:
            comparison, target = instruction.split(':')
            field = comparison[0]
            op = operator.lt if comparison[1] == "<" else operator.gt
            value = int(comparison[2:])
            return Rule(field=field, operator=op, value=value, if_true=target)
        return Rule(if_true=instruction)

    def parse_workflow(self, line: str) -> Workflow:
        name, rules_string = line.split('{', maxsplit=2)
        rules = list(map(self.parse_rule, rules_string[:-1].split(',')))
        return Workflow(name=name, rules=rules)

    def parse_part(self, line: str) -> Part:
        return Part(*map(int, re.findall(r"\d+", line)))

    def preprocess(self) -> tuple[dict[str, Workflow], list[Part]]:
        workflows_string, parts_string = self.separated_by_empty_line()
        workflows = {w.name: w for w in map(self.parse_workflow, workflows_string.split("\n"))}
        parts = list(map(self.parse_part, parts_string.split("\n")))
        return workflows, parts

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 19114)], result=425811, dataset_class=DataSet)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 167409079868000)], result=None,
                                  dataset_class=DataSet)
# pytest_generate_tests = generate_rounds(round_1, round_2)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    workflows, parts = dataset.preprocess()
    assert sum(map(partial(workflows['in'].rating, workflows=workflows), parts)) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert dataset.result is None
