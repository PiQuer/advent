"""
--- Day 18: Lavaduct Lagoon ---
https://adventofcode.com/2023/day/18
"""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import pytest
import tinyarray as ta
from more_itertools import peekable

from adventofcode.utils import dataset_parametrization, DataSetBase, cross_product

YEAR= "2023"
DAY= "18"

HEADINGS = {
    "R": ta.array((1, 0)),
    "D": ta.array((0, -1)),
    "L": ta.array((-1, 0)),
    "U": ta.array((0, 1))
}


@dataclass(frozen=True)
class Segment:
    start: ta.ndarray_int
    stop: ta.ndarray_int
    incoming_heading: ta.ndarray_int
    outgoing_heading: ta.ndarray_int

    def area(self, orientation: int) -> float:
        return orientation * ((self.start[0] - self.stop[0]) * self.start[1] +
                              cross_product(self.incoming_heading, self.outgoing_heading) / 4.) + \
            sum(ta.abs(self.stop - self.start)) / 2. \


class Loop:
    def __init__(self, segments: list[Segment]):
        self._segments = segments
        self._orientation = -1 if sum(cross_product(s.incoming_heading, s.outgoing_heading) for s in segments) < 0 \
            else 1

    def area(self):
        return int(sum(s.area(self._orientation) for s in self._segments))


class DataSetDay18(DataSetBase, metaclass=ABCMeta):
    @abstractmethod
    def line_to_params(self, line:str) -> tuple[ta.ndarray_int, int]:
        pass

    def loop(self) -> Loop:
        result: list[Segment] = []
        lines = peekable(self.line_to_params(line) for line in self.lines())
        start = ta.array((0, 0))
        for incoming_heading, length in lines:
            next_line = lines.peek(None)
            outgoing_heading = result[0].incoming_heading if next_line is None else next_line[0]
            stop = start + length * incoming_heading
            result.append(Segment(start=start, stop=stop, incoming_heading=incoming_heading,
                                  outgoing_heading=outgoing_heading))
            start = stop
        return Loop(result)

class DataSetRound1(DataSetDay18):
    def line_to_params(self, line:str) -> tuple[ta.ndarray_int, int]:
        direction, length, _ = line.split()
        return HEADINGS[direction], int(length)

class DataSetRound2(DataSetDay18):
    def line_to_params(self, line:str) -> tuple[ta.ndarray_int, int]:
        directive = line.split()[2]
        return list(HEADINGS.values())[int(directive[-2:-1])], int(directive[2:-2], 16)

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 62)], result=62500, dataset_class=DataSetRound1)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 952408144115)], result=122109860712709,
                                  dataset_class=DataSetRound2)

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetRound1):
    loop = dataset.loop()
    assert loop.area() == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetRound2):
    loop = dataset.loop()
    assert loop.area() == dataset.result
