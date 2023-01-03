import pytest

from utils import dataset_parametrization, DataSetBase


number_of_segments = {0: 6, 1: 2, 2: 5, 3: 5, 4: 4, 5: 5, 6: 6, 7: 3, 8: 7, 9: 6}
unique_segments = (2, 3, 4, 7)


class DataSet(DataSetBase):
    def get_data(self):
        return [[list(map(set, x.split())) for x in line.split('|')] for line in self.lines()]
    
    
round_1 = dataset_parametrization("2021", "08", [("", 26)], result=445, dataset_class=DataSet)
round_2 = dataset_parametrization("2021", "08", [("", 61229)], result=1043101, dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_part_one(dataset: DataSet):
    data = dataset.get_data()
    uniques = [w for line in data for w in line[1] if len(w) in unique_segments]
    assert len(uniques) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_part_two(dataset: DataSet):
    data = dataset.get_data()
    result = 0
    for signal, display in data:
        p = {}
        for digit in range(10):
            p[digit] = [s for s in signal if number_of_segments[digit] == len(s)]
            if len(p[digit]) == 1:
                p[digit] = p[digit][0]
        p[5] = [s for s in p[5] if p[4]-p[1] < s][0]
        p[0] = [s for s in p[0] if not p[4]-p[1] < s][0]
        p[2] = [s for s in p[2] if p[0]-p[4] < s][0]
        p[3] = [s for s in p[3] if p[1] < s][0]
        p[6] = [s for s in p[6] if not p[1] < s][0]
        p[9] = [s for s in p[9] if p[3] < s][0]
        inv_p = {tuple(sorted(v)): k for k, v in p.items()}
        result += int("{}{}{}{}".format(*[inv_p[tuple(sorted(d))] for d in display]))
    assert result == dataset.result
