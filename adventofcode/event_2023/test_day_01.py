"""
--- Day 1: Trebuchet?! ---
https://adventofcode.com/2023/day/1
"""
import re

from adventofcode.utils import DataSetBase, dataset_parametrization, generate_parts


class DataSet(DataSetBase):
    translation = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7',
                   'eight': '8', 'nine': '9'}
    def preprocessed_lines(self):
        digits = r'|'.join(self.translation.keys())
        yield from (re.sub(rf'(?=({digits}))', lambda match: self.translation[match[1]], line)
                    for line in self.lines())

    def calibration_values(self, lines_function):
        yield from (int((d := ''.join(filter(str.isdigit, line)))[0] + d[-1]) for line in \
                    lines_function(self))


round_1 = dataset_parametrization(year="2023", day="01", examples=[("1", 142)], part=1, dataset_class=DataSet,
                                  lines=DataSet.lines)
round_2 = dataset_parametrization(year="2023", day="01", examples=[("2", 281)], part=2, dataset_class=DataSet,
                                  lines=DataSet.preprocessed_lines)
pytest_generate_tests = generate_parts(round_1, round_2)


def test_day_1(dataset: DataSet):
    dataset.assert_answer(sum(dataset.calibration_values(dataset.params["lines"])))
