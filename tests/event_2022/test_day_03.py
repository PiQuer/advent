import pytest
from utils import dataset_parametrization


round_1 = dataset_parametrization(year="2022", day="03", examples=[("", 157)], result=7568)
round_2 = dataset_parametrization(year="2022", day="03", examples=[("", 70)], result=2780)


def priority(c):
    return ord(c) - 38 if c < 'a' else ord(c) - 96


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset):
    priority_sum = 0
    for line in dataset.lines():
        half = len(line) // 2
        priority_sum += priority((set(line[:half]) & set(line[half:])).pop())
    assert priority_sum == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset):
    priority_sum = 0
    data = dataset.lines()
    for g in range(0, len(data), 3):
        priority_sum += priority((set(data[g]) & set(data[g+1]) & set(data[g+2])).pop())
    assert priority_sum == dataset.result
