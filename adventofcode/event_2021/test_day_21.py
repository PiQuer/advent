"""
--- Day 21: Dirac Dice ---
https://adventofcode.com/2021/day/21
"""
import itertools
from collections import defaultdict, Counter

import numpy as np
import pytest

from adventofcode.utils import dataset_parametrization, DataSetBase

deterministic_die = 100
board = 10
points_to_win_det = 1000
points_to_win_dirac = 21


class DataSet(DataSetBase):
    def get_start_positions(self):
        return tuple(int(i.split(': ')[1]) for i in self.lines())


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


round_1 = dataset_parametrization("2021", "21", [("", 739785)], result=921585, dataset_class=DataSet)
round_2 = dataset_parametrization("2021", "21", [("", 444356092776315)], result=911090395997650, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_part_one(dataset: DataSet):
    start_positions = dataset.get_start_positions()
    assert play_deterministic(*start_positions) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_part_two(dataset: DataSet):
    start_positions = dataset.get_start_positions()
    assert play_dirac(*start_positions) == dataset.result
