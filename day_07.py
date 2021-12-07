from typing import Callable
import numpy as np


def crab_align(input_file: str, fuel_cost: Callable[[np.ndarray], np.ndarray]):
    positions = np.genfromtxt(input_file, delimiter=',', dtype=int)
    abs_distances = np.abs(positions - np.arange(np.max(positions) + 1).reshape(-1, 1))
    min_fuel = np.min(np.sum(fuel_cost(abs_distances).astype(int), axis=1))
    print(f"The optimal horizontal position can be reached with {min_fuel} fuel spent.")


if __name__ == '__main__':
    [crab_align(f, fuel_cost=fuel_cost) for fuel_cost in (lambda x: x, lambda x: x*(x+1)/2)
     for f in ("input/day_07_example.txt", "input/day_07.txt")]
