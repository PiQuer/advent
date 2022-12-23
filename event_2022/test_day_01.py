import pytest
from utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def calories(self):
        yield from map(lambda x: sum(int(s) for s in x.split()), self.separated_by_empty_line())


round_1 = dataset_parametrization(day="01", examples=[("", 24000)], result=66186, dataset_class=DataSet, top=1)
round_2 = dataset_parametrization(day="01", examples=[("", 45000)], result=196804, dataset_class=DataSet, top=3)


# noinspection PyMethodMayBeStatic
class Base:
    def test_with_sort(self, dataset: DataSet):
        return sum(sorted(dataset.calories())[-dataset.params["top"]:])


@pytest.mark.parametrize(**round_1)
class TestRound1(Base):
    pass


@pytest.mark.parametrize(**round_2)
class TestRound2(Base):
    pass
