from itertools import accumulate

import pytest
from collections import deque

from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(day="07", examples=[("", 95437)], result=1350966)
round_2 = dataset_parametrization(day="07", examples=[("", 24933642)], result=6296435)


def build_tree(lines):
    sizes = {'/': 0}
    current = deque(('/',))
    for line in lines[1:]:
        if line == "$ ls":
            continue
        if line == "$ cd ..":
            current.pop()
        elif line.startswith("$ cd "):
            current.append(f"{line[5:]}/")
        elif line.startswith("dir "):
            sizes["".join(current) + f"{line[4:]}/"] = 0
        else:
            size, name = line.split(' ')
            size = int(size)
            for d in accumulate(current):
                sizes[d] += size
    return sizes


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    sizes = build_tree(dataset.lines())
    assert sum(v for v in sizes.values() if v <= 100000) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    sizes = build_tree(dataset.lines())
    missing = 30000000 - (70000000 - sizes['/'])
    assert min(v for v in sizes.values() if v >= missing) == dataset.result
