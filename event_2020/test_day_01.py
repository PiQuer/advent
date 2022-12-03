import pytest
from pathlib import Path
from itertools import tee, filterfalse, product, combinations, chain


input_files_round_1 = (
    (Path("input/day_01_example.txt"), 514579),
    (Path("input/day_01.txt"), 270144),
)


input_files_round_2 = (
    (Path("input/day_01_example.txt"), 241861950),
    (Path("input/day_01.txt"), 261342720),
)


def partition(pred, iterable, make_list=False):
    t1, t2 = tee(iterable)
    r1, r2 = filterfalse(pred, t1), filter(pred, t2)
    return (list(r1), list(r2)) if make_list else (r1, r2)


@pytest.mark.parametrize("input_file,expected", input_files_round_1)
def test_round_1(input_file, expected):
    larger, smaller = partition(lambda x: x < 1010, (int(line) for line in input_file.read_text().splitlines()))
    n1 = n2 = 0
    for n1, n2 in product(larger, smaller):
        if n1 + n2 == 2020:
            break
    assert n1 * n2 == expected


@pytest.mark.parametrize("input_file,expected", input_files_round_2)
def test_round_2(input_file, expected):
    lower, rest = partition(lambda x: x > 2020//3, (int(line) for line in input_file.read_text().splitlines()),
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
    assert n1 * n2 * n3 == expected
