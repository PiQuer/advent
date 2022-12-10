import pytest
from typing import Iterator
from itertools import islice, count
from operator import mul
import numpy as np
from more_itertools import chunked

from utils import dataset_parametrization, DataSetBase


example_output = """##..##..##..##..##..##..##..##..##..##..
###...###...###...###...###...###...###.
####....####....####....####....####....
#####.....#####.....#####.....#####.....
######......######......######......####
#######.......#######.......#######....."""
result = """####...##.#..#.###..#..#.#....###..####.
#.......#.#..#.#..#.#..#.#....#..#....#.
###.....#.#..#.###..#..#.#....#..#...#..
#.......#.#..#.#..#.#..#.#....###...#...
#....#..#.#..#.#..#.#..#.#....#.#..#....
#.....##...##..###...##..####.#..#.####."""


class DataSet(DataSetBase):
    def preprocess(self) -> Iterator[int]:
        x = 1
        for line in self.lines():
            if line.startswith("noop"):
                yield x
            elif line.startswith("addx"):
                yield from iter((x, x))
                x += int(line[5:])


round_1 = dataset_parametrization(day="10", examples=[("", 13140)], result=13060, dataset_class=DataSet)
round_2 = dataset_parametrization(day="10", examples=[("", example_output)], result=result, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    start, stop, step = 20, 220, 40
    assert sum(islice((mul(*x) for x in zip(count(1), dataset.preprocess())), start-1, stop, step)) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    width = 40
    crt = np.fromiter((sprite-1 <= crt % width <= sprite+1 for crt, sprite in zip(count(), dataset.preprocess())),
                      dtype=np.bool)
    display = '\n'.join(''.join(line) for line in chunked(np.where(crt, b'#', b'.').tobytes().decode(), width))
    print("")
    print(display)
    assert display == dataset.result
