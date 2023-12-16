"""
--- Day 2: Rock Paper Scissors ---
https://adventofcode.com/2022/day/2
"""
from pathlib import Path

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_rounds

FILE_NAMES = {'example': Path("input/day_02_example.txt"), 'real': Path("input/day_02.txt")}


class DataSet(DataSetBase):
    def lines(self):
        for line in super().lines():
            yield ord(line[0]) - 65, ord(line[2]) - 88


round_1 = dataset_parametrization(year="2022", day="02", examples=[("", 15)], result=10310, dataset_class=DataSet,
                                  outcome=False)
round_2 = dataset_parametrization(year="2022", day="02", examples=[("", 12)], result=14859, dataset_class=DataSet,
                                  outcome=True)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_day_2(dataset: DataSet):
    score = 0
    for p1, p2 in dataset.lines():
        if dataset.params["outcome"]:
            p2 = (p1 + p2 - 1) % 3
        score += p2 + 1 + ((p2 - p1 + 1) % 3) * 3
    assert score == dataset.result
