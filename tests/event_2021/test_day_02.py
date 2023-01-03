import numpy as np
import pytest

from utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def get_data(self):
        return np.genfromtxt(self.input_file, dtype=[('dir', '|S2'), ('amount', '<i8')])


round_1 = dataset_parametrization(year="2021", day="02", examples=[("", 150)], result=2187380, dataset_class=DataSet)
round_2 = dataset_parametrization(year="2021", day="02", examples=[("", 900)], result=2086357770,
                                  dataset_class=DataSet)


def get_masks(data):
    return map(lambda x: data['dir'] == x, (b'fo', b'up', b'do'))


@pytest.mark.parametrize(**round_1)
def test_part_one(dataset: DataSet):
    data = dataset.get_data()
    forward_mask, up_mask, down_mask = get_masks(data)
    horizontal = data['amount'][forward_mask].sum()
    depth = data['amount'][down_mask].sum() - data['amount'][up_mask].sum()
    assert horizontal * depth == dataset.result


@pytest.mark.parametrize(**round_2)
def test_part_two(dataset: DataSet):
    data = dataset.get_data()
    aim_change = np.zeros_like(data, dtype=int)
    forward = np.zeros_like(data, dtype=int)
    forward_mask, up_mask, down_mask = get_masks(data)
    aim_change[up_mask] -= data['amount'][up_mask]
    aim_change[down_mask] += data['amount'][down_mask]
    aim = np.cumsum(aim_change)
    forward[forward_mask] += data['amount'][forward_mask]
    horizontal = forward.sum()
    depth = np.dot(forward, aim)
    assert horizontal * depth == dataset.result
