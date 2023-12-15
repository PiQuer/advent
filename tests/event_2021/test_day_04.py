"""
--- Day 4: Giant Squid ---
https://adventofcode.com/2021/day/4
"""
from io import StringIO
import numpy as np
import pytest

from utils import dataset_parametrization, DataSetBase


board_size = 5


class DataSet(DataSetBase):
    def get_data(self):
        data = self.text()
        numbers_str, _, boards_str = data.split("\n", 2)
        numbers = np.fromstring(numbers_str, sep=',', dtype=int)
        boards = np.reshape(np.genfromtxt(StringIO(boards_str), dtype=int), (-1, board_size, board_size))
        return numbers, boards


round_1_and_2 = dataset_parametrization(year="2021", day="04", examples=[("", (4512, 1924))],
                                        result=(55770, 2980), dataset_class=DataSet)


def get_score(boards_to_consider, boards, marked, n):
    assert boards_to_consider.sum() == 1, "Ambiguous solution."
    return boards[boards_to_consider][~marked[boards_to_consider]].sum() * n


@pytest.mark.parametrize(**round_1_and_2)
def test_bingo(dataset: DataSet):
    numbers, boards = dataset.get_data()
    marked = np.zeros_like(boards, dtype=bool)
    winning = np.zeros(boards.shape[0], dtype=bool)
    result_part_one, result_part_two = None, None
    for n in numbers:
        marked |= boards == n
        now_winning = np.any((marked.sum(axis=2) == board_size) | (marked.sum(axis=1) == board_size), axis=1)
        winning_in_this_round = np.logical_xor(winning, now_winning)
        if np.any(winning_in_this_round) and np.all(~winning):
            result_part_one = get_score(winning_in_this_round, boards, marked, n)
            print(f"First winning score: {result_part_one}")
        if np.any(winning_in_this_round) and np.all(now_winning):
            result_part_two = get_score(winning_in_this_round, boards, marked, n)
            print(f"Last winning score: {result_part_two}")
            break
        winning = now_winning
    assert result_part_one == dataset.result[0]
    assert result_part_two == dataset.result[1]
