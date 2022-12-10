import pytest
from datetime import datetime
import re
from collections import Counter, defaultdict
from itertools import chain, repeat
from operator import mul

from utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_asleep = defaultdict(lambda: 0)

    def sleep_times(self):
        self.total_asleep = defaultdict(lambda: 0)
        current_id = None
        falls_asleep = None
        for line in sorted(self.lines()):
            if m := re.search(r"#(\d+) begins shift", line):
                current_id = int(m.group(1))
                continue
            time = datetime.strptime(line[1:17], "%Y-%m-%d %H:%M")
            if "falls asleep" in line:
                falls_asleep = time
            else:
                self.total_asleep[current_id] += int((time - falls_asleep).total_seconds()) // 60
                yield current_id, falls_asleep, time


round_1 = dataset_parametrization(day="04", examples=[("", 240)], result=143415, dataset_class=DataSet)
round_2 = dataset_parametrization(day="04", examples=[("", 4455)], result=49944, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    c = Counter(chain.from_iterable(zip(repeat(g_id), range(a.minute, w.minute))
                                    for g_id, a, w in dataset.sleep_times()))
    sleepiest_guard = max(dataset.total_asleep.keys(), key=lambda i: dataset.total_asleep[i])
    sleepiest_minute = max((k for k in c.keys() if k[0] == sleepiest_guard), key=lambda k: c[k])
    assert mul(*sleepiest_minute) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    c = Counter(chain.from_iterable(zip(repeat(g_id), range(a.minute, w.minute))
                                    for g_id, a, w in dataset.sleep_times()))
    assert mul(*max((m for m in c.keys()), key=lambda m: c[m])) == dataset.result
