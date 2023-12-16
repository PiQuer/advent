"""
--- Day 7: The Treachery of Whales ---
https://adventofcode.com/2021/day/7
"""
import numpy as np
from toolz import identity

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_rounds


round_1 = dataset_parametrization("2021", "07", [("", 37)], result=356922, fuel_cost=identity)
round_2 = dataset_parametrization("2021", "07", [("", 168)], result=100347031, fuel_cost=lambda x: x*(x+1)/2)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_crab_align(dataset: DataSetBase):
    fuel_cost = dataset.params["fuel_cost"]
    positions = np.genfromtxt(dataset.input_file, delimiter=',', dtype=int)
    abs_distances = np.abs(positions - np.arange(np.max(positions) + 1).reshape(-1, 1))
    assert np.min(np.sum(fuel_cost(abs_distances).astype(int), axis=1)) == dataset.result
