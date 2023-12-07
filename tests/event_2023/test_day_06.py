import math
from dataclasses import dataclass
from functools import reduce
from operator import mul

from utils import dataset_parametrization, DataSetBase, generate_rounds


@dataclass
class Race:
    time: int
    distance: int

    def ways_to_win(self):
        sqrt_d = math.sqrt(self.time**2/4 - self.distance)
        return math.ceil(self.time/2 + sqrt_d - 1) - math.floor(self.time/2 - sqrt_d + 1) + 1


class DataSet1(DataSetBase):
    def races(self) -> list[Race]:
        times = map(int, self.lines()[0].split()[1:])
        distances = map(int, self.lines()[1].split()[1:])
        return list(Race(*args) for args in zip(times, distances))


class DataSet2(DataSetBase):
    def races(self) -> list[Race]:
        time = int(self.lines()[0].split(maxsplit=1)[1].replace(" ", ""))
        distance = int(self.lines()[1].split(maxsplit=1)[1].replace(" ", ""))
        return [Race(time, distance)]


round_1 = dataset_parametrization(year="2023", day="06", examples=[("", 288)], result=741000,
                                  dataset_class=DataSet1)
round_2 = dataset_parametrization(year="2023", day="06", examples=[("", 71503)], result=38220708,
                                  dataset_class=DataSet2)
pytest_generate_tests = generate_rounds(round_1, round_2)

def test_day_6(dataset: DataSet1|DataSet2):
    assert reduce(mul, (race.ways_to_win() for race in dataset.races())) == dataset.result
