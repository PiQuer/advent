"""
--- Day XX: XXXXX ---
https://adventofcode.com/2023/day/XX
"""
from dataclasses import dataclass

import pytest
import tinyarray as ta
from more_itertools import peekable

from adventofcode.utils import dataset_parametrization, DataSetBase, cross_product

# from adventofcode.utils import generate_rounds

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
            else -1

    def area(self):
        return int(sum(s.area(self._orientation) for s in self._segments))

class DataSet(DataSetBase):
    def loop(self) -> Loop:
        result: list[Segment] = []
        lines = peekable(line.split() for line in self.lines())
        start = ta.array((0, 0))
        for direction, length, _ in lines:
            next_line = lines.peek(None)
            incoming_heading = HEADINGS[direction]
            outgoing_heading = result[0].incoming_heading if next_line is None else HEADINGS[next_line[0]]
            stop = start + int(length) * incoming_heading
            result.append(Segment(start=start, stop=stop, incoming_heading=incoming_heading,
                                  outgoing_heading=outgoing_heading))
            start = stop
        return Loop(result)

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 62)], result=62500, dataset_class=DataSet)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", None)], result=None, dataset_class=DataSet)
# pytest_generate_tests = generate_rounds(round_1, round_2)

@pytest.mark.parametrize(
    "segment, expected",
    ((Segment(start=ta.array((0, 0)), stop=ta.array((1, 0)),
              incoming_heading=ta.array((1, 0)), outgoing_heading=ta.array((0, 1))), 0.75),
     (Segment(start=ta.array((1, 0)), stop=ta.array((1, 1)),
              incoming_heading=ta.array((0, 1)), outgoing_heading=ta.array((-1, 0))), 0.75),
     (Segment(start=ta.array((1, 1)), stop=ta.array((0, 1)),
              incoming_heading=ta.array((-1, 0)), outgoing_heading=ta.array((0, -1))), 1.75),
     (Segment(start=ta.array((0, 1)), stop=ta.array((0, 0)),
              incoming_heading=ta.array((0, -1)), outgoing_heading=ta.array((1, 0))), 0.75))
)
def test_area(segment: Segment, expected: float):
    assert segment.area() == expected


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    loop = dataset.loop()
    assert loop.area() == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert dataset.result is None
