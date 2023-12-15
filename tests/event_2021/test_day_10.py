"""
--- Day 10: Syntax Scoring ---
https://adventofcode.com/2021/day/10
"""
import pytest
from pathlib import Path

from utils import dataset_parametrization, DataSetBase


delimiters = {'(': ')', '[': ']', '{': '}', '<': '>'}


def get_data(input_file: str):
    return Path(input_file).read_text().splitlines()


class IllegalCharacter(Exception):
    scores = {')': 3, ']': 57, '}': 1197, '>': 25137}

    def __init__(self, character):
        super().__init__()
        self.score = self.scores[character]


class IncompleteLineException(Exception):
    scores = {')': 1, ']': 2, '}': 3, '>': 4}

    def __init__(self, stack):
        super().__init__()
        self.score = 0
        for c in reversed(stack):
            self.score = self.score*5 + self.scores[c]


def parse_chunk(line):
    stack = ''
    for char in line:
        if char in delimiters.keys():
            stack += delimiters[char]
        elif char in delimiters.values():
            if char == stack[-1]:
                stack = stack[:-1]
            else:
                raise IllegalCharacter(char)
    if stack:
        raise IncompleteLineException(stack)


round_1_2 = dataset_parametrization("2021", "10", [("", (26397, 288957))], result=(345441, 3235371166))


@pytest.mark.parametrize(**round_1_2)
def test_syntax_scoring(dataset: DataSetBase):
    illegal_character_score = 0
    incomplete_line_scores = []
    for line in dataset.lines():
        try:
            parse_chunk(line.strip())
        except IllegalCharacter as e:
            illegal_character_score += e.score
        except IncompleteLineException as e:
            incomplete_line_scores.append(e.score)
    assert illegal_character_score == dataset.result[0]
    incomplete_line_score = sorted(incomplete_line_scores)[int(len(incomplete_line_scores)/2)]
    assert incomplete_line_score == dataset.result[1]
