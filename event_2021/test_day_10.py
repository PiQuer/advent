import pytest
from pathlib import Path


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


@pytest.mark.parametrize("input_file", ["input/day_10_example.txt", "input/day_10.txt"])
class TestDay10:
    def test_syntax_scoring(self, input_file):
        illegal_character_score = 0
        incomplete_line_scores = []
        for line in get_data(input_file):
            try:
                parse_chunk(line.strip())
            except IllegalCharacter as e:
                illegal_character_score += e.score
            except IncompleteLineException as e:
                incomplete_line_scores.append(e.score)
        assert illegal_character_score == (26397 if "example" in input_file else 345441)
        incomplete_line_score = sorted(incomplete_line_scores)[int(len(incomplete_line_scores)/2)]
        assert incomplete_line_score == (288957 if "example" in input_file else 3235371166)
