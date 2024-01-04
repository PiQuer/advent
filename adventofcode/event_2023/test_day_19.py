"""
--- Day 19: Aplenty ---
https://adventofcode.com/2023/day/19
"""
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial, reduce

import pytest
import tinyarray as ta

from adventofcode.utils import dataset_parametrization, DataSetBase

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
class Constraint:
    lower: list[int] = field(default_factory=lambda: [1]*4)
    upper: list[int] = field(default_factory=lambda: [4001]*4)

    def valid(self) -> bool:
        return all(e > 0 for e in ta.array(self.upper) - ta.array(self.lower))

    def combinations(self) -> int:
        return reduce(operator.mul, ta.array(self.upper) - ta.array(self.lower), 1)

INDEX = {"x": 0, "m": 1, "a": 2, "s": 3, None: None}

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

    def combinations(self, constraint: Constraint, workflows: dict[str, "Workflow"]) -> int:
        result = 0
        for rule in self.rules:
            idx = INDEX[rule.field]
            next_constraint_true = Constraint(lower=constraint.lower.copy(), upper=constraint.upper.copy())
            if rule.operator is operator.lt:
                next_constraint_true.upper[idx] = min(next_constraint_true.upper[idx], rule.value)
                constraint.lower[idx] = max(constraint.lower[idx], rule.value)
            elif rule.operator is operator.gt:
                next_constraint_true.lower[idx] = max(next_constraint_true.lower[idx], rule.value + 1)
                constraint.upper[idx] = min(constraint.upper[idx], rule.value + 1)
            if next_constraint_true.valid():
                match rule.if_true:
                    case 'A':
                        result += next_constraint_true.combinations()
                    case 'R':
                        result += 0
                    case next_workflow:
                        result += workflows[next_workflow].combinations(constraint=next_constraint_true,
                                                                        workflows=workflows)
        return result

class DataSet(DataSetBase):
    def parse_rule(self, instruction: str) -> Rule:
        if ':' in instruction:
            comparison, target = instruction.split(':')
            f = comparison[0]
            op = operator.lt if comparison[1] == "<" else operator.gt
            value = int(comparison[2:])
            return Rule(field=f, operator=op, value=value, if_true=target)
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
        parts = list(map(self.parse_part, parts_string.strip().split("\n")))
        return workflows, parts

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 19114)], dataset_class=DataSet, part=1)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 167409079868000)], dataset_class=DataSet, part=2)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    workflows, parts = dataset.preprocess()
    dataset.assert_answer(sum(map(partial(workflows['in'].rating, workflows=workflows), parts)))


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    workflows, _ = dataset.preprocess()
    dataset.assert_answer(workflows['in'].combinations(Constraint(), workflows=workflows))
