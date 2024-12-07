"""
--- Day 1: The Tyranny of the Rocket Equation ---
https://adventofcode.com/2019/day/1
"""
from functools import partial

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_parts


def calculate_fuel(mass: int, recursive: bool):
    fuel = mass // 3 - 2
    if not recursive:
        return fuel
    return 0 if fuel < 0 else fuel + calculate_fuel(mass=fuel, recursive=True)


round_1 = dataset_parametrization("2019", "01", [], result=3398090, recursive=False)
round_2 = dataset_parametrization("2019", "01", [], result=5094261, recursive=True)
pytest_generate_tests = generate_parts(round_1, round_2)


def test_day_01(dataset: DataSetBase):
    assert sum(map(partial(calculate_fuel, recursive=dataset.params["recursive"]), dataset.np_array())) == \
           dataset.result
