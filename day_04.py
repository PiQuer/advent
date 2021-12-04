from io import StringIO
from pathlib import Path
import numpy as np


board_size = 5


def get_data(input_file: str):
    data = Path(input_file).read_text()
    numbers_str, _, boards_str = data.split("\n", 2)
    numbers = np.fromstring(numbers_str, sep=',', dtype=int)
    boards = np.reshape(np.genfromtxt(StringIO(boards_str), dtype=int), (-1, board_size, board_size))
    return numbers, boards


def get_winning_board(mask):
    return np.argwhere(np.bitwise_or(mask.sum(axis=2) == board_size, mask.sum(axis=1) == board_size))


def bingo(input_file: str):
    numbers, boards = get_data(input_file)
    mask = np.zeros_like(boards, dtype=bool)
    n = 0
    winning = None
    for n in numbers:
        mask = np.bitwise_or(mask, boards == n)
        winning = get_winning_board(mask)
        if winning.size > 0:
            break
    assert winning is not None, "No winner was found."
    assert winning.shape[0] == 1, "Multiple winners were found."
    winning_board = winning[0, 0]
    result = boards[winning_board][~mask[winning_board]].sum() * n
    print(f"Score: {result}")


if __name__ == '__main__':
    bingo("input/day_04_example.txt")
    bingo("input/day_04.txt")
