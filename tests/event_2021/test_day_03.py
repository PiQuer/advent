import numpy as np
import pytest

from utils import dataset_parametrization, DataSetBase


def bin_to_int(data: np.array):
    return data.dot(2**np.arange(data.size)[::-1])


class DataSet(DataSetBase):
    def rating(self, invert=False) -> int:
        data = self.np_array_digits()
        column = 0
        while data.shape[0] > 1 and column < data.shape[1]:
            mask = data[:, column] == (data[:, column].sum() >= data.shape[0]/2)
            if invert:
                mask = np.invert(mask)
            data = data[mask]
            column += 1
        assert data.shape[0] == 1, "This puzzle has no unique solution."
        return bin_to_int(data.squeeze(axis=0))


round_1 = dataset_parametrization(year="2021", day="03", examples=[("", 198)], result=4191876, dataset_class=DataSet)
round_2 = dataset_parametrization(year="2021", day="03", examples=[("", 230)], result=3414905, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_power_consumption(dataset: DataSet):
    data = dataset.np_array_digits()
    gamma = np.round(np.sum(data, axis=0) / data.shape[0]).astype(bool)
    epsilon = ~gamma
    assert bin_to_int(epsilon) * bin_to_int(gamma) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_ratings(dataset: DataSet):
    oxygen_generator_rating = dataset.rating()
    co2_scrubber_rating = dataset.rating(invert=True)
    assert oxygen_generator_rating * co2_scrubber_rating == dataset.result
