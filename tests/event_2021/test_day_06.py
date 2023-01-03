import numpy as np

from utils import dataset_parametrization, DataSetBase, generate_rounds


binsize = 9
delay = 2


round_1 = dataset_parametrization("2021", "06", [("", 5934)], result=386536, number_of_days=80)
round_2 = dataset_parametrization("2021", "06", [("", 26984457539)], result=1732821262171, number_of_days=256)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_lanternfish(dataset: DataSetBase):
    number_of_days = dataset.params["number_of_days"]
    timer_counts = np.bincount(np.genfromtxt(dataset.input_file, delimiter=',', dtype=int), minlength=binsize)
    for day in range(number_of_days):
        timer_counts[binsize-delay] += timer_counts[0]
        timer_counts = np.roll(timer_counts, -1)
    assert timer_counts.sum() == dataset.result
