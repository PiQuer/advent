import numpy as np


def get_data(input_file: str):
    return np.genfromtxt(input_file, dtype=[('dir', '|S2'), ('amount', '<i8')])


def get_masks(data):
    return map(lambda x: data['dir'] == x, (b'fo', b'up', b'do'))


def part_one(input_file: str):
    data = get_data(input_file)
    forward_mask, up_mask, down_mask = get_masks(data)
    horizontal = data['amount'][forward_mask].sum()
    depth = data['amount'][down_mask].sum() - data['amount'][up_mask].sum()
    print(f"Result part one: {horizontal * depth}")


def part_two(input_file: str):
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
    print(f"Result part two: {horizontal * depth}")


if __name__ == '__main__':
    part_one("input/day_02_example.txt")
    part_one("input/day_02.txt")
    part_two("input/day_02_example.txt")
    part_two("input/day_02.txt")
