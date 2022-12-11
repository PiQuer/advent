import numpy as np
import pytest


def get_data(input_file: str):
    return np.genfromtxt(input_file, dtype=[('dir', '|S2'), ('amount', '<i8')])


def get_masks(data):
    return map(lambda x: data['dir'] == x, (b'fo', b'up', b'do'))


@pytest.mark.parametrize("input_file", ["input/day_02_example.txt", "input/day_02.txt"])
class TestDay02:
    def test_part_one(self, input_file: str):
        data = get_data(input_file)
        forward_mask, up_mask, down_mask = get_masks(data)
        horizontal = data['amount'][forward_mask].sum()
        depth = data['amount'][down_mask].sum() - data['amount'][up_mask].sum()
        result = horizontal * depth
        print(f"Result part one: {result}")
        assert result == (150 if "example" in input_file else 2187380)

    def test_part_two(self, input_file: str):
        data = get_data(input_file)
        aim_change = np.zeros_like(data, dtype=int)
        forward = np.zeros_like(data, dtype=int)
        forward_mask, up_mask, down_mask = get_masks(data)
        aim_change[up_mask] -= data['amount'][up_mask]
        aim_change[down_mask] += data['amount'][down_mask]
        aim = np.cumsum(aim_change)
        forward[forward_mask] += data['amount'][forward_mask]
        horizontal = forward.sum()
        depth = np.dot(forward, aim)
        result = horizontal * depth
        print(f"Result part two: {result}")
        assert result == (900 if "example" in input_file else 2086357770)
