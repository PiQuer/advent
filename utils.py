from dataclasses import dataclass, field
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
        return self.input_file.read_text().splitlines()


    def text(self) -> str:
        return self.input_file.read_text()

    def separated_by_empty_line(self) -> list[str]:
        return self.input_file.read_text().split("\n\n")

    def np_array(self, dtype=np.int32) -> np.array:
        return np.loadtxt(self.input_file, dtype=dtype)

    def np_array_digits(self) -> np.array:
        return np.genfromtxt(self.input_file, dtype=int, delimiter=1)

    def np_array_bytes(self) -> np.array:
        x = np.array(self.lines(), dtype=bytes)
        return x.view('S1').reshape((x.size, -1))

    def np_array_str(self) -> np.array:
        return np.array([list(line) for line in self.lines()])


# noinspection PyArgumentList
def dataset_parametrization(year: str, day: str, examples: Sequence[tuple[Any, ...]], result: Any,
                            dataset_class: type[DataSetBase] = DataSetBase, **kwargs):
    current_dir = Path(__file__).parent
    base_dir = current_dir / "tests" / f"event_{year}" / "input"
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


def adjacent_3d() -> Iterator[tuple[int, ...]]:
    return (a for a in adjacent_with_diag_3d() if sum(map(abs, a)) == 1)


def np_grid() -> Iterator[np.array]:
    return (np.array(a) for a in grid())


def np_adjacent_with_diag() -> Iterator[np.array]:
    return (np.array(a) for a in adjacent_with_diag())


def np_adjacent() -> Iterator[np.array]:
    return (np.array(a) for a in adjacent())


def ta_adjacent(pos: ta.array = ta.array((0, 0))) -> Iterator[ta.array]:
    return (pos + ta.array(a) for a in adjacent())


def ta_adjacent_with_diag_3d() -> Iterator[ta.array]:
    return map(ta.array, adjacent_with_diag_3d())


def ta_adjacent_3d() -> Iterator[ta.array]:
    return map(ta.array, adjacent_3d())


def directions() -> dict[str, tuple[int, ...]]:
    return dict(zip(('l', 'u', 'd', 'r'), adjacent()))


def np_directions() -> dict[str, np.array]:
    return dict(zip(('l', 'u', 'd', 'r'), np_adjacent()))


def ta_directions(chars=('l', 'u', 'd', 'r')) -> dict[str, ta.array]:
    return dict(zip(chars, ta_adjacent()))


def ta_directions_arrows() -> dict[str, ta.array]:
    return ta_directions(chars=('^', '<', '>', 'v'))


@dataclass
class Waypoint:
    length: Union[int, float]
    previous: Optional[Union[tuple[int, int], ta.array]]
    value: Optional[Any] = None


def generate_rounds(round_1: dict, round_2: dict):
    def pytest_generate_tests_template(metafunc):
        if "dataset" in metafunc.fixturenames:
            metafunc.parametrize("dataset", argvalues=round_1['argvalues'] + round_2['argvalues'],
                                 ids=lambda x: ("round1_" if x in round_1['argvalues'] else "round2_") + x.id)
    return pytest_generate_tests_template


def inbounds(shape: ta.array, pos: ta.array) -> bool:
    return max(pos - shape) < 0 <= min(pos)
