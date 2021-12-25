import numpy as np


def shift(array: np.array, amount=1, axis=0, fill=11):
    result = np.roll(np.copy(array), amount, axis=axis)
    index = [slice(None) for _ in result.shape]
    index[axis] = slice(min(np.sign(amount), 0), amount + min(np.sign(amount), 0), np.sign(amount))
    result[tuple(index)] = fill
    return result
