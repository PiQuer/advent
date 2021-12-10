from io import StringIO
from pathlib import Path
import numpy as np
import pytest


board_size = 5


def get_data(input_file: str):
    data = Path(input_file).read_text()
    numbers_str, _, boards_str = data.split("\n", 2)
    numbers = np.fromstring(numbers_str, sep=',', dtype=int)
    boards = np.reshape(np.genfromtxt(StringIO(boards_str), dtype=int), (-1, board_size, board_size))
    return numbers, boards


def get_score(boards_to_consider, boards, marked, n):
    assert boards_to_consider.sum() == 1, "Ambiguous solution."
    return boards[boards_to_consider][~marked[boards_to_consider]].sum() * n


@pytest.mark.parametrize("input_file", ["input/day_04_example.txt", "input/day_04.txt"])
def test_bingo(input_file: str):
    numbers, boards = get_data(input_file)
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
    assert result_part_one == (4512 if "example" in input_file else 55770)
    assert result_part_two == (1924 if "example" in input_file else 2980)
