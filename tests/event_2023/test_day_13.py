import numpy as np
import pytest

from utils import dataset_parametrization, DataSetBase

year="2023"
day="13"

class DataSet(DataSetBase):
    def preprocess(self):
        for lines in self.separated_by_empty_line():
            x = np.array(lines.split(), dtype=bytes)
            yield x.view('S1').reshape((x.size, -1))


round_1 = dataset_parametrization(year=year, day=day, examples=[("", 405)], result=33780, dataset_class=DataSet)
round_2 = dataset_parametrization(year=year, day=day, examples=[("", 400)], result=23479, dataset_class=DataSet)


def get_mirror_axis_horizontal(a: np.ndarray, smudge: bool = False) -> int | None:
    if smudge:
        candidates = np.argwhere(np.sum(np.logical_not(a[:-1] == a[1:]), axis=1) <= 1).reshape((-1,))
    else:
        candidates = np.argwhere(np.all(a[:-1] == a[1:], axis=1)).reshape((-1,))
    for c in candidates:
        mid = c + 1
        if mid <= a.shape[0] // 2:
            upper, lower = 2 * mid, 0
        else:
            upper, lower = a.shape[0], 2 * mid - a.shape[0]
        if np.sum(np.logical_not(a[lower:mid] == np.flip(a[mid:upper], axis=0))) == (1 if smudge else 0):
            return mid
    return None


def get_mirror_axis(a: np.ndarray) -> int:
    return (100 * h) if (h := get_mirror_axis_horizontal(a)) is not None else get_mirror_axis_horizontal(a.transpose())

def get_smudge_axis(a: np.ndarray) -> int:
    r = (100 * h) if (h := get_mirror_axis_horizontal(a, smudge=True)) is not None else \
        get_mirror_axis_horizontal(a.transpose(), smudge=True)
    return r


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    assert sum(map(get_mirror_axis, dataset.preprocess())) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert (sum(map(get_smudge_axis, dataset.preprocess()))) == dataset.result
