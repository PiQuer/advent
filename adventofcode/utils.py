"""Useful helpers."""
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from pathlib import Path
from typing import Sequence, Any, Union, Optional, Iterator

import numpy as np
import tinyarray as ta


@dataclass
class DataSetBase:
    input_file: Path
    result: Any
    id: str
    params: dict[Any, Any] = field(default_factory=dict)

    def lines(self) -> list[str]:
        return self.input_file.read_text(encoding="ascii").splitlines()


    def text(self) -> str:
        return self.input_file.read_text(encoding="ascii")

    def bytes(self) -> bytes:
        return self.input_file.read_bytes()

    def separated_by_empty_line(self) -> list[str]:
        return self.input_file.read_text(encoding="ascii").split("\n\n")

    def np_array(self, dtype=np.int32) -> np.array:
        return np.loadtxt(self.input_file, dtype=dtype)

    def np_array_digits(self) -> np.ndarray:
        return np.genfromtxt(self.input_file, dtype=int, delimiter=1)

    @cached_property
    def np_array_bytes(self) -> np.ndarray:
        x = np.array(self.lines(), dtype=bytes)
        return x.view('S1').reshape((x.size, -1))

    def np_array_str(self) -> np.ndarray:
        return np.array([list(line) for line in self.lines()])


# noinspection PyArgumentList
def dataset_parametrization(year: str, day: str, examples: Sequence[tuple[Any, ...]], result: Any,
                            dataset_class: type[DataSetBase] = DataSetBase, **kwargs):
    current_dir = Path(__file__).parent
    base_dir = current_dir / f"event_{year}" / "input"
    examples = [dataset_class(
        input_file=base_dir/f"day_{day}_example{example[0]}.txt",
        result=example[1],
        id=f"example{example[0]}",
        params=dict(kwargs, **(example[-1] if len(example) == 3 else {})))
        for example in examples]
    puzzle = dataset_class(input_file=base_dir/f"day_{day}.txt", result=result, id="puzzle", params=kwargs)
    return {'argnames': "dataset", 'argvalues': examples + [puzzle], 'ids': lambda x: x.id}


def grid() -> Iterator[tuple[int, ...]]:
    return (a for a in product((-1, 0, 1), repeat=2))


def grid_3d() -> Iterator[tuple[int, ...]]:
    return (a for a in product((-1, 0, 1), repeat=3))


def adjacent_with_diag() -> Iterator[tuple[int, ...]]:
    return (a for a in grid() if a != (0, 0))


def adjacent_with_diag_3d() -> Iterator[tuple[int, ...]]:
    return (a for a in grid_3d() if a != (0, 0, 0))


def adjacent() -> Iterator[tuple[int, ...]]:
    return (a for a in adjacent_with_diag() if abs(a[0]) != abs(a[1]))


def ta_adjacent() -> Iterator[ta.ndarray_int]:
    return (ta.array(a) for a in adjacent())

def adjacent_3d() -> Iterator[tuple[int, ...]]:
    return (a for a in adjacent_with_diag_3d() if sum(map(abs, a)) == 1)


def np_grid() -> Iterator[np.ndarray]:
    return (np.array(a) for a in grid())


def np_adjacent_with_diag() -> Iterator[np.ndarray]:
    return (np.array(a) for a in adjacent_with_diag())


def np_adjacent() -> Iterator[np.ndarray]:
    return (np.array(a) for a in adjacent())


def ta_adjacent(pos: ta.array = ta.array((0, 0))) -> Iterator[ta.ndarray_int]:
    return (pos + ta.array(a) for a in adjacent())


def ta_adjacent_with_diag_3d() -> Iterator[ta.ndarray_int]:
    return map(ta.array, adjacent_with_diag_3d())


def ta_adjacent_3d() -> Iterator[ta.ndarray_int]:
    return map(ta.array, adjacent_3d())


def directions() -> dict[str, tuple[int, ...]]:
    return dict(zip(('l', 'u', 'd', 'r'), adjacent()))


def np_directions() -> dict[str, np.ndarray]:
    return dict(zip(('l', 'u', 'd', 'r'), np_adjacent()))


def ta_directions(chars=('l', 'u', 'd', 'r')) -> dict[str, ta.ndarray_int]:
    return dict(zip(chars, ta_adjacent()))


def ta_directions_arrows() -> dict[str, ta.ndarray_int]:
    return ta_directions(chars=('^', '<', '>', 'v'))


@dataclass
class Waypoint:
    length: Union[int, float]
    previous: Optional[Union[tuple[int, int], ta.ndarray_int]]
    value: Optional[Any] = None


def generate_rounds(round_1: dict, round_2: dict):
    def pytest_generate_tests_template(metafunc):
        if "dataset" in metafunc.fixturenames:
            metafunc.parametrize("dataset", argvalues=round_1['argvalues'] + round_2['argvalues'],
                                 ids=lambda x: ("round1_" if x in round_1['argvalues'] else "round2_") + x.id)
    return pytest_generate_tests_template


def inbounds(shape: ta.ndarray_int, pos: ta.ndarray_int) -> bool:
    return max(pos - shape) < 0 <= min(pos)


def shift(array: np.ndarray, amount=1, axis=0, fill=11):
    result = np.roll(np.copy(array), amount, axis=axis)
    index = [slice(None) for _ in result.shape]
    index[axis] = slice(min(np.sign(amount), 0), amount + min(np.sign(amount), 0), np.sign(amount))
    result[tuple(index)] = fill
    return result


def cross_product(v1: ta.ndarray_int, v2: ta.ndarray_int) -> int:
    return v1[0]*v2[1] - v1[1]*v2[0]
