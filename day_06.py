import numpy as np


timer_max = 8
delay = 2


def lanternfish(input_file: str, number_of_days: int):
    timer_counts = np.bincount(np.genfromtxt(input_file, delimiter=',', dtype=int), minlength=timer_max + 1)
    for day in range(number_of_days):
        births = timer_counts[0]
        timer_counts[:-1] = timer_counts[1:]
        timer_counts[timer_max] = births
        timer_counts[timer_max - delay] += births
    print(f"There are {timer_counts.sum()} fish after {number_of_days} days.")


if __name__ == '__main__':
    lanternfish("input/day_06_example.txt", number_of_days=80)
    lanternfish("input/day_06.txt", number_of_days=80)
    lanternfish("input/day_06_example.txt", number_of_days=256)
    lanternfish("input/day_06.txt", number_of_days=256)
