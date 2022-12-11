from typing import Sequence, Any, Iterator
from pathlib import Path
from dataclasses import dataclass
from itertools import product
import numpy as np
import tinyarray as ta


@dataclass
class DataSetBase:
    input_file: Path
    result: Any
    id: str

    def lines(self) -> list[str]:
        return self.input_file.read_text().splitlines()

    def text(self) -> str:
        return self.input_file.read_text()

    def separated_by_empty_line(self):
        return self.input_file.read_text().split("\n\n")

    def np_array(self, dtype=np.int32) -> np.array:
        return np.loadtxt(self.input_file, dtype=dtype)

    def np_array_digits(self) -> np.array:
        return np.genfromtxt(self.input_file, dtype=int, delimiter=1)

    def np_array_bytes(self) -> np.array:
        x = np.array(self.lines(), dtype=bytes)
        return x.view('S1').reshape((x.size, -1))


# noinspection PyArgumentList
def dataset_parametrization(day: str, examples: Sequence[tuple[str, Any]], result: Any,
                            dataset_class: type[DataSetBase] = DataSetBase):
    examples = [dataset_class(input_file=Path(f"input/day_{day}_example{example[0]}.txt"),
                              result=example[1],
                              id=f"example{example[0]}") for example in examples]
    puzzle = dataset_class(input_file=Path(f"input/day_{day}.txt"), result=result, id="puzzle")
    return {'argnames': "dataset", 'argvalues': examples + [puzzle], 'ids': lambda x: x.id}


def grid() -> Iterator[tuple[int, int]]:
    return (a for a in product((-1, 0, 1), repeat=2))


def adjacent_with_diag() -> Iterator[tuple[int, int]]:
    return (a for a in grid() if a != (0, 0))


def adjacent() -> Iterator[tuple[int, int]]:
    return (a for a in adjacent_with_diag() if abs(a[0]) != abs(a[1]))


def np_grid() -> Iterator[np.array]:
    return (np.array(a) for a in grid())


def np_adjacent_with_diag() -> Iterator[np.array]:
    return (np.array(a) for a in adjacent_with_diag())


def np_adjacent() -> Iterator[np.array]:
    return (np.array(a) for a in adjacent())


def ta_adjacent() -> Iterator[ta.array]:
    return (ta.array(a) for a in adjacent())


def directions() -> dict[str, tuple[int, int]]:
    return dict(zip(('l', 'u', 'd', 'r'), adjacent()))


def np_directions() -> dict[str, np.array]:
    return dict(zip(('l', 'u', 'd', 'r'), np_adjacent()))


def ta_directions() -> dict[str, ta.array]:
    return dict(zip(('l', 'u', 'd', 'r'), ta_adjacent()))
