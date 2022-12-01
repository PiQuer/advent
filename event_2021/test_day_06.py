import numpy as np
import pytest


binsize = 9
delay = 2


@pytest.mark.parametrize("input_file", ["input/day_06_example.txt", "input/day_06.txt"])
@pytest.mark.parametrize("number_of_days", [80, 256])
def test_lanternfish(input_file: str, number_of_days: int):
    timer_counts = np.bincount(np.genfromtxt(input_file, delimiter=',', dtype=int), minlength=binsize)
    for day in range(number_of_days):
        timer_counts[binsize-delay] += timer_counts[0]
        timer_counts = np.roll(timer_counts, -1)
    result = timer_counts.sum()
    print(f"There are {result} fish after {number_of_days} days.")
    example = "example" in input_file
    assert result == {
        (True, 80): 5934,
        (True, 256): 26984457539,
        (False, 80): 386536,
        (False, 256): 1732821262171}[(example, number_of_days)]
