"""
https://adventofcode.com/2022/day/17
"""
import pytest
import tinyarray as ta
import numpy as np
from itertools import cycle, islice
from dataclasses import dataclass
from collections import deque
import logging

from more_itertools import always_reversible

from utils import dataset_parametrization, DataSetBase


@dataclass
class Rock:
    sprite: ta.array
    height: int
    width: int

    @property
    def shape(self):
        return self.height, self.width


rocks = (
    Rock(sprite=ta.array(((1, 1, 1, 1),)), height=1, width=4),
    Rock(sprite=ta.array(((0, 1, 0), (1, 1, 1), (0, 1, 0))), height=3, width=3),
    Rock(sprite=ta.array(((1, 1, 1), (0, 0, 1), (0, 0, 1))), height=3, width=3),
    Rock(sprite=ta.array(((1,), (1,), (1,), (1,))), height=4, width=1),
    Rock(sprite=ta.array(((1, 1), (1, 1))), height=2, width=2)
)


round_1 = dataset_parametrization(day="17", examples=[("", 3068)], result=3137)
round_2 = dataset_parametrization(day="17", examples=[("", 1_514_285_714_288)], result=1_564_705_882_327)


def visualize(c: np.array):
    for row in np.array(c)[::-1]:
        print(''.join('#' if x else ' ' for x in row))


class CycleDetector:
    def __init__(self, threshold):
        self._threshold = threshold
        self._deque = deque()
        self._cycle_length = 0
        self._start = None

    def check(self, dh: int, idx: int, current_height: int):
        self._deque.appendleft(dh)
        if len(self._deque) > self._threshold:
            if self._start is None:
                self._start = list(islice(self._deque, 1, None))
            if list(islice(self._deque, self._threshold)) == self._start:
                raise EndOfProblem(
                    list(always_reversible(islice(self._deque, len(self._deque) - self._threshold))), idx,
                    current_height)


class EndOfProblem(Exception):
    pass


class BaseDay17:
    rounds: int

    def test_day_17(self, dataset: DataSetBase):
        jet_it = map(lambda x: {'>': 1, '<': -1}[x], cycle(dataset.text()))
        rock_it = cycle(rocks)
        c = np.ones((1, 7), dtype=int)
        cycle_detector = CycleDetector(1000)
        last_height = 0
        try:
            for idx, rock in enumerate(rock_it):
                if idx % 1000 == 0:
                    logging.info("Index: %d", idx)
                row = np.argwhere(np.max(c, axis=1))[-1][0] + 3 + rock.height
                col = 2
                if (append := row - c.shape[0] + 1) > 0:
                    c = np.concatenate([c, np.zeros((append, c.shape[1]), dtype=int)], axis=0)
                for jet in jet_it:
                    if 0 <= col + jet <= c.shape[1] - rock.width:
                        if not np.any(
                                np.logical_and(c[row-rock.height+1:row+1, col+jet:col+jet+rock.width], rock.sprite)):
                            col += jet
                    if not np.any(np.logical_and(c[row-rock.height:row, col:col+rock.width], rock.sprite)):
                        row -= 1
                    else:
                        c[row - rock.height + 1:row + 1, col:col + rock.width] += rock.sprite
                        dh = (height := np.argwhere(np.max(c, axis=1))[-1][0]) - last_height
                        last_height = height
                        if idx > 1000:
                            cycle_detector.check(dh, idx, height)
                        break
        except EndOfProblem as e:
            h_cycle, idx, current_height = e.args
            cycles, rest = divmod(self.rounds - idx - 1, len(h_cycle))
            height = current_height + sum(h_cycle) * cycles + sum(islice(h_cycle, rest))
        else:
            assert False
        assert height == dataset.result


@pytest.mark.parametrize(**round_1)
class TestRound1(BaseDay17):
    rounds = 2022


@pytest.mark.parametrize(**round_2)
class TestRound2(BaseDay17):
    rounds = 1_000_000_000_000
