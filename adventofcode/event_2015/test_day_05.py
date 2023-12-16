"""
--- Day 5: Doesn't He Have Intern-Elves For This? ---
https://adventofcode.com/2015/day/5
"""
import re

from toolz import count

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_rounds


class DataSet(DataSetBase):
    def nice_lines_1(self):
        for line in self.lines():
            if re.search(r"([aeiou].*){3}", line) and re.search(r"(.)\1", line) and \
                    (re.search(r"ab|cd|pq|xy", line) is None):
                yield line

    def nice_lines_2(self):
        for line in self.lines():
            if re.search(r"(..).*\1", line) and re.search(r"(.).\1", line):
                yield line


round_1 = dataset_parametrization("2015", "05", examples=[("1", 2)], result=236, dataset_class=DataSet,
                                  fn=DataSet.nice_lines_1)
round_2 = dataset_parametrization("2015", "05", examples=[("2", 2)], result=51, dataset_class=DataSet,
                                  fn=DataSet.nice_lines_2)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_day_05(dataset: DataSet):
    assert count(dataset.params["fn"](dataset)) == dataset.result
