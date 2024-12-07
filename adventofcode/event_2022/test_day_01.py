"""
--- Day 1: Calorie Counting ---
https://adventofcode.com/2022/day/1
"""
import heapq

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_parts


class DataSet(DataSetBase):
    def calories(self):
        yield from map(lambda x: sum(int(s) for s in x.split()), self.separated_by_empty_line())


round_1 = dataset_parametrization(year="2022", day="01", examples=[("", 24000)], result=66186, dataset_class=DataSet,
                                  top=max)
round_2 = dataset_parametrization(year="2022", day="01", examples=[("", 45000)], result=196804, dataset_class=DataSet,
                                  top=lambda x: sum(heapq.nlargest(3, x)))
pytest_generate_tests = generate_parts(round_1, round_2)


def test_day_1(dataset: DataSet):
    assert dataset.params["top"](dataset.calories()) == dataset.result
