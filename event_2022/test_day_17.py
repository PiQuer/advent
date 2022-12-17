"""
https://adventofcode.com/2022/day/17
"""
from operator import itemgetter

import pytest
import tinyarray as ta
import numpy as np
from itertools import cycle, islice
from dataclasses import dataclass
from collections import deque
import math
import logging
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
round_2 = dataset_parametrization(day="17", examples=[("", 1_514_285_714_288)], result=None)


def visualize(c: np.array):
    for row in np.array(c)[::-1]:
        print(''.join('#' if x else ' ' for x in row))


class CycleDetector:
    threshold = 2 * len(rocks)
    max_cyc = 20

    def __init__(self, lcm: int):
        self._lcm = lcm
        self._deques = {}
        for i in range(1, self.max_cyc + 1):
            self._deques[i] = deque()

    def check(self, state: tuple, idx: int):
        for i in range(1, self.max_cyc + 1):
            self._deques[i].append(state)
            if len(self._deques[i]) > i * self._lcm + self.threshold:
                self._deques[i].popleft()
                if list(map(itemgetter(slice(0, 2)), list(islice(self._deques[i], self.threshold)))) == \
                        list(map(itemgetter(slice(0, 2)),
                                 list(reversed(list(islice(reversed(self._deques[i]), self.threshold)))))):
                    raise EndOfProblem(self._deques[i], idx, i*self._lcm)
        pass


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    jet_it = map(lambda x: {'>': 1, '<': -1}[x], cycle(dataset.text()))
    rock_it = cycle(rocks)
    c = np.ones((1, 7), dtype=int)
    total_removed = 0
    for idx, rock in enumerate(islice(rock_it, 2022)):
        if idx % 100 == 99:
            removed, c = housekeeping(c)
            total_removed += removed
        row = np.argwhere(np.max(c, axis=1))[-1][0] + 3 + rock.height
        col = 2
        if (append := row - c.shape[0] + 1) > 0:
            c = np.concatenate([c, np.zeros((append, c.shape[1]), dtype=int)], axis=0)
        for jet in jet_it:
            if 0 <= col + jet <= c.shape[1] - rock.width:
                if not np.any(np.logical_and(c[row-rock.height+1:row+1, col+jet:col+jet+rock.width], rock.sprite)):
                    col += jet
            if not np.any(np.logical_and(c[row-rock.height:row, col:col+rock.width], rock.sprite)):
                row -= 1
            else:
                c[row - rock.height + 1:row + 1, col:col + rock.width] += rock.sprite
                break
    assert np.argwhere(np.max(c, axis=1))[-1][0] + total_removed == dataset.result


class EndOfProblem(Exception):
    pass


def housekeeping(c: np.array) -> tuple[int, np.array]:
    cut_row = np.argwhere(np.min(np.add.accumulate(np.flipud(c), axis=0), axis=1) > 5)[0][0] + 1
    result = c.shape[0] - cut_row
    return result, np.delete(c, np.s_[:-cut_row], 0)


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    jet_it = map(lambda x: {'>': 1, '<': -1}[x], cycle(dataset.text()))
    rock_it = cycle(enumerate(rocks))
    c = np.ones((1, 7), dtype=int)
    total_removed = 0
    lcm = math.lcm(len(rocks), len(dataset.text()))
    cycle_detector = CycleDetector(lcm)
    logging.info("LCM: %d", lcm)
    try:
        for idx, (rock_idx, rock) in enumerate(rock_it):
            if idx % (lcm//10) == 0:
                logging.info("Index, Index/LCM: %d, %f", idx, idx/lcm)
            if idx % 100 == 99:
                removed, c = housekeeping(c)
                total_removed += removed
            row = np.argwhere(np.max(c, axis=1))[-1][0] + 3 + rock.height
            col = 2
            if (append := row - c.shape[0] + 1) > 0:
                c = np.concatenate([c, np.zeros((append, c.shape[1]), dtype=int)], axis=0)
            for jet in jet_it:
                if 0 <= col + jet <= c.shape[1] - rock.width:
                    if not np.any(np.logical_and(c[row-rock.height+1:row+1, col+jet:col+jet+rock.width], rock.sprite)):
                        col += jet
                if not np.any(np.logical_and(c[row-rock.height:row, col:col+rock.width], rock.sprite)):
                    row -= 1
                else:
                    c[row - rock.height + 1:row + 1, col:col + rock.width] += rock.sprite
                    cycle_detector.check((rock_idx, col, np.argwhere(np.max(c, axis=1))[-1][0] + total_removed), idx)
                    break
    except EndOfProblem as e:
        d, idx, cyc = e.args
        cyc_height = d[-1][2] - d[-1-cyc][2]
        cycles, cycles_rest = divmod(1_000_000_000_000 - idx, cyc)
        height = d[-1][2] + cycles * cyc_height + d[-cyc-1+cycles_rest][2] - d[-cyc][2]
    else:
        assert False
    assert height == dataset.result
