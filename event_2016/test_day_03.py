import numpy as np


def test_round_1():
    data = np.sort(np.genfromtxt("input/day_03.txt", dtype=np.int32), axis=1)
    assert np.sum(data[:, 0] + data[:, 1] > data[:, 2]) == 983


def test_round_2():
    data = np.sort(np.genfromtxt("input/day_03.txt", dtype=np.int32).transpose().reshape(-1, 3), axis=1)
    assert np.sum(data[:, 0] + data[:, 1] > data[:, 2]) == 1836
