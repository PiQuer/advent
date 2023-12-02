import dataclasses
import re
from typing import Generator

import pytest

from utils import dataset_parametrization, DataSetBase


@dataclasses.dataclass
class Subset:
    red: int = 0
    green: int = 0
    blue: int = 0

    def power(self) -> int:
        return self.red * self.green * self.blue

    def __le__(self, other):
        return self.red <= other.red and self.green <= other.green and self.blue <= other.blue

@dataclasses.dataclass
class Game:
    id: int
    subsets: tuple[Subset]

    def min_subset(self) -> Subset:
        return Subset(**{color: max(s.__getattribute__(color) for s in self.subsets)
                         for color in ("red", "green", "blue")})


class DataSet(DataSetBase):
    @staticmethod
    def subset_parser(subset: str) -> Subset:
        return Subset(**{match[2]: int(match[1]) for match in re.finditer(r"(\d+) (red|green|blue)", subset)})

    def game_parser(self, line: str) -> Game:
        game_id = int(re.match(r"Game (\d+):", line)[1])
        return Game(id=game_id, subsets=tuple(self.subset_parser(subset) for subset in line.split(';')))

    def preprocess(self) -> Generator[Game, None, None]:
        yield from (self.game_parser(line) for line in self.lines())



round_1 = dataset_parametrization(year="2023", day="02", examples=[("", 8)], result=2237, dataset_class=DataSet)
round_2 = dataset_parametrization(year="2023", day="02", examples=[("", 2286)], result=66681, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    bag = Subset(red=12, green=13, blue=14)
    assert sum(game.id for game in dataset.preprocess() if all(g <= bag for g in game.subsets)) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert sum(game.min_subset().power() for game in dataset.preprocess()) == dataset.result