import pytest
from itertools import tee, filterfalse, product, combinations, chain

from utils import dataset_parametrization, DataSetBase


def partition(pred, iterable, make_list=False):
    t1, t2 = tee(iterable)
    r1, r2 = filterfalse(pred, t1), filter(pred, t2)
    return (list(r1), list(r2)) if make_list else (r1, r2)


round_1 = dataset_parametrization("2020", "01", [("", 514579)], result=270144)
round_2 = dataset_parametrization("2020", "01", [("", 241861950)], result=261342720)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    larger, smaller = partition(lambda x: x < 1010, (int(line) for line in dataset.lines()))
    n1 = n2 = 0
    for n1, n2 in product(larger, smaller):
        if n1 + n2 == 2020:
            break
    assert n1 * n2 == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    lower, rest = partition(lambda x: x > 2020//3, (int(line) for line in dataset.lines()),
                            make_list=True)
    middle, upper = partition(lambda x: x > 1010, rest, make_list=True)
    for n1, n2, n3 in chain(
            ((c1, c2, c3) for (c1, c2), c3 in product(combinations(lower, 2), rest)),
            ((c1, c2, c3) for c1, (c2, c3) in product(lower, combinations(middle, 2))),
            product(lower, middle, upper)
    ):
        if n1 + n2 + n3 == 2020:
            break
    else:
        assert False
    assert n1 * n2 * n3 == dataset.result
