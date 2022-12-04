import pytest
from collections import defaultdict
from itertools import product
from pathlib import Path


class Intcode:
    def __init__(self, initial: str):
        self.reg = defaultdict(lambda: 0)
        for i, c in enumerate(initial.split(',')):
            self.reg[i] = int(c)
        self._s = 0

    def compute(self):
        while self.reg[self._s] != 99:
            if self.reg[self._s] == 1:
                self.reg[self.reg[self._s + 3]] = self.reg[self.reg[self._s + 1]] + self.reg[self.reg[self._s + 2]]
            elif self.reg[self._s] == 2:
                self.reg[self.reg[self._s + 3]] = self.reg[self.reg[self._s + 1]] * self.reg[self.reg[self._s + 2]]
            else:
                assert False
            self._s += 4

    def output(self):
        return ','.join(f"{self.reg[i]}" for i in range(max(self.reg.keys()) + 1))


@pytest.mark.parametrize(
    "input_string,output_string",
    (("1,9,10,3,2,3,11,0,99,30,40,50", "3500,9,10,70,2,3,11,0,99,30,40,50"),
     ("1,0,0,0,99", "2,0,0,0,99"),
     ("2,3,0,3,99", "2,3,0,6,99"),
     ("2,4,4,5,99,0", "2,4,4,5,99,9801"),
     ("1,1,1,4,99,5,6,0,99", "30,1,1,4,2,5,6,0,99"))
)
def test_intcode(input_string, output_string):
    i = Intcode(input_string)
    i.compute()
    assert i.output() == output_string


def test_round_1():
    i = Intcode(Path("input/day_02.txt").read_text())
    i.reg[1] = 12
    i.reg[2] = 2
    i.compute()
    assert i.reg[0] == 4090701


def test_round_2():
    init = Path("input/day_02.txt").read_text()
    for noun, verb in product(range(100), range(100)):
        i = Intcode(init)
        i.reg[1], i.reg[2] = noun, verb
        i.compute()
        if i.reg[0] == 19690720:
            break
    else:
        assert False
    assert 100*noun + verb == 6421
