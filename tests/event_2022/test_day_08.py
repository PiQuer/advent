import pytest
import numpy as np

from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(year="2022", day="08", examples=[("", 21)], result=1688)
round_2 = dataset_parametrization(year="2022", day="08", examples=[("", 8)], result=410400)


def visible(heights: np.array):
    left = np.maximum.accumulate(heights[1:-1, :-2], axis=1)
    up = np.maximum.accumulate(heights[:-2, 1:-1], axis=0)
    return (heights[1:-1, 1:-1] > left) | (heights[1:-1, 1:-1] > up)


def sight(heights: np.array):
    result = np.zeros_like(heights[1:-1, 1:-1])
    for y, x in np.ndindex(*result.shape):
        result[y, x] = next((c+1 for c, h in enumerate(heights[y+1, x+2:]) if heights[y+1, x+1] <= h),
                            heights.shape[1] - 2 - x)
    return result


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    heights = dataset.np_array_digits()
    assert 2*sum(heights.shape) - 4 + np.sum(visible(heights) | np.flip(visible(np.flip(heights)))) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    heights = dataset.np_array_digits()
    assert np.max(sight(heights) * np.rot90(sight(np.rot90(heights, axes=(1, 0)))) *
                  np.flip(sight(np.flip(heights))) * np.rot90(sight(np.rot90(heights)), axes=(1, 0))) == dataset.result
