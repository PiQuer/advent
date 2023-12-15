"""
--- Day 5: Supply Stacks ---
https://adventofcode.com/2022/day/5
"""
from collections import defaultdict
from itertools import takewhile, islice

from utils import dataset_parametrization, DataSetBase, generate_rounds


class DataSet(DataSetBase):
    def preprocessed_lines(self):
        i = iter(self.lines())
        stack_dict = defaultdict(lambda: [])
        for line in reversed(list(takewhile(lambda x: x[:2] != ' 1', i))):
            for num, char in enumerate(islice(line, 1, None, 4)):
                if char != ' ':
                    stack_dict[num+1].append(char)
        yield stack_dict
        next(i)  # consume empty line
        for line in i:
            yield tuple(int(c) for c in line.split(' ') if c.isdigit())


round_1 = dataset_parametrization("2022", "05", examples=[("", "CMZ")], result="SHMSDGZVC", dataset_class=DataSet,
                                  slice=lambda num: slice(-1, -(num+1), -1))
round_2 = dataset_parametrization("2022", "05", examples=[("", "MCD")], result="VRZGHDFBQ", dataset_class=DataSet,
                                  slice=lambda num: slice(-num, None))
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_day_5(dataset: DataSet):
    it = dataset.preprocessed_lines()
    stack = next(it)
    for num, f, t in it:
        stack[t].extend(stack[f][dataset.params['slice'](num)])
        del(stack[f][-num:])
    assert ''.join(stack[k][-1] for k in stack.keys()) == dataset.result
