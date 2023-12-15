"""
--- Day 6: Tuning Trouble ---
https://adventofcode.com/2022/day/6
"""
from utils import dataset_parametrization, DataSetBase, generate_rounds


round_1 = dataset_parametrization(year="2022", day="06", examples=[("1", 7), ("2", 5), ("3", 6), ("4", 10), ("5", 11)],
                                  result=1794, num_unique=4)
round_2 = dataset_parametrization(year="2022", day="06",
                                  examples=[("1", 19), ("2", 23), ("3", 23), ("4", 29), ("5", 26)],
                                  result=2851, num_unique=14)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_day_6(dataset: DataSetBase):
    input_string = dataset.text()
    num_unique = dataset.params['num_unique']
    assert next(k for k in range(num_unique, len(input_string))
                if len(set(input_string[k-num_unique:k])) == num_unique) == dataset.result
