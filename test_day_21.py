import pytest
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import itertools

deterministic_die = 100
board = 10
points_to_win_det = 1000
points_to_win_dirac = 21


def get_start_positions(input_file):
    data = Path(input_file).read_text().splitlines()
    return tuple(int(i.split(': ')[1]) for i in data)


def play_deterministic(pos_1, pos_2):
    points = np.array([0, 0])
    pos = np.array([pos_1, pos_2]) - 1
    d = 0
    winner_found = False
    while not winner_found:
        for p in (0, 1):
            pos[p] = (pos[p] + (np.arange(d, d+3) % deterministic_die + 1).sum()) % board
            points[p] += pos[p] + 1
            d += 3
            if points[p] >= points_to_win_det:
                winner_found = True
                break
    return d * min(points)


def play_dirac(pos_1, pos_2):
    outcomes = Counter([sum(i) for i in itertools.product((1, 2, 3), (1, 2, 3), (1, 2, 3))])
    winning = np.array([0, 0])
    states = defaultdict(lambda: 0)
    states[(pos_1-1, pos_2-1, 0, 0)] = 1
    active = 0
    while states:
        states_next = defaultdict(lambda: 0)
        for state, universes in states.items():
            for outcome, new_universes in outcomes.items():
                pos, points = np.array(state[0:2]), np.array(state[2:4])
                pos[active] += outcome
                pos[active] %= board
                points[active] += pos[active] + 1
                if points[active] >= points_to_win_dirac:
                    winning[active] += universes * new_universes
                else:
                    states_next[tuple(pos) + tuple(points)] += universes * new_universes
        states = states_next
        active = 1 - active
    return max(winning)


@pytest.mark.parametrize("input_file,expected",
                         (("input/day_21_example.txt", 739785),
                          ("input/day_21.txt", 921585)))
def test_part_one(input_file, expected):
    start_positions = get_start_positions(input_file)
    result = play_deterministic(*start_positions)
    assert result == expected


@pytest.mark.parametrize("input_file,expected",
                         (("input/day_21_example.txt", 444356092776315),
                          ("input/day_21.txt", 911090395997650)))
def test_part_two(input_file, expected):
    start_positions = get_start_positions(input_file)
    result = play_dirac(*start_positions)
    assert result == expected
