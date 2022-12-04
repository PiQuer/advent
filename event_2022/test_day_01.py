import pytest
from utils import dataset_parametrization, DataSetBase


class DataSet(DataSetBase):
    def preprocess(self):
        current = 0
        for line in self.input_file.read_text().split('\n'):
            if not line:
                yield current
                current = 0
            else:
                current += int(line)


round_1 = dataset_parametrization(day="01", examples=[("", 24000)], result=66186, dataset_class=DataSet)
round_2 = dataset_parametrization(day="01", examples=[("", 45000)], result=196804, dataset_class=DataSet)


@pytest.mark.benchmark(warmup=True, warmup_iterations=100)
class Base:
    def test_with_sort(self, dataset: DataSet):
        return sum(sorted(dataset.preprocess())[-self.top:])


@pytest.mark.parametrize(**round_1)
class TestRound1(Base):
    top = 1


@pytest.mark.parametrize(**round_2)
class TestRound2(Base):
    top = 3
