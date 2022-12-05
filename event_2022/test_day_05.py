import pytest
from collections import defaultdict
from itertools import takewhile, islice
from utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def preprocess(self):
        i = iter(super().preprocess())
        stack_dict = defaultdict(lambda: [])
        for line in reversed(list(takewhile(lambda x: x[:2] != ' 1', i))):
            for num, char in enumerate(islice(line, 1, None, 4)):
                if char != ' ':
                    stack_dict[num+1].append(char)
        yield stack_dict
        next(i)  # consume empty line
        for line in i:
            yield tuple(int(c) for c in line.split(' ') if c.isdigit())


round_1 = dataset_parametrization("05", examples=[("", "CMZ")], result="SHMSDGZVC", dataset_class=DataSet)
round_2 = dataset_parametrization("05", examples=[("", "MCD")], result="VRZGHDFBQ", dataset_class=DataSet)


class Day05Base:
    model = None

    def test_restack(self, dataset: DataSet):
        it = dataset.preprocess()
        stack = next(it)
        for num, f, t in it:
            if self.model == 9000:
                stack[t].extend(stack[f][-1:-(num+1):-1])
            else:
                stack[t].extend(stack[f][-num:])
            del(stack[f][-num:])
        assert ''.join(stack[k][-1] for k in stack.keys()) == dataset.result


@pytest.mark.parametrize(**round_1)
class TestRound1(Day05Base):
    model = 9000


@pytest.mark.parametrize(**round_2)
class TestRound2(Day05Base):
    model = 9001
