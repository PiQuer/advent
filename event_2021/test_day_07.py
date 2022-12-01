import numpy as np
import pytest


@pytest.mark.parametrize("input_file", ["input/day_07_example.txt", "input/day_07.txt"])
@pytest.mark.parametrize("fuel_cost", [lambda x: x, lambda x: x*(x+1)/2])
def test_crab_align(input_file, fuel_cost):
    positions = np.genfromtxt(input_file, delimiter=',', dtype=int)
    abs_distances = np.abs(positions - np.arange(np.max(positions) + 1).reshape(-1, 1))
    min_fuel = np.min(np.sum(fuel_cost(abs_distances).astype(int), axis=1))
    print(f"The optimal horizontal position can be reached with {min_fuel} fuel spent.")
    example = "example" in input_file
    linear = fuel_cost(2) == 2
    assert min_fuel == \
           {(True, True): 37, (True, False): 168, (False, True): 356922, (False, False): 100347031}[(example, linear)]
