"""
--- Day 14: Extended Polymerization ---
https://adventofcode.com/2021/day/14
"""
from collections import defaultdict

from utils import dataset_parametrization, DataSetBase, generate_rounds


class DataSet(DataSetBase):
    def get_data(self):
        mapping = {}
        with self.input_file.open('r') as f:
            template_str = f.readline().strip() + " "
            f.readline()
            for line in f.readlines():
                key, value = line.strip().split(' -> ')
                mapping[key] = value
        template = defaultdict(lambda: 0)
        for n in range(len(template_str) - 1):
            template[template_str[n:n+2]] += 1
        return template, mapping


def do_step(polymer, mapping):
    result = polymer.copy()
    for key in polymer.keys():
        insertion = mapping.get(key)
        if insertion is not None:
            result[key[0]+insertion] += polymer[key]
            result[insertion+key[1]] += polymer[key]
            result[key] -= polymer[key]
    return result


def count(polymer):
    result = defaultdict(lambda: 0)
    for key, value in polymer.items():
        result[key[0]] += value
    return result


round_1 = dataset_parametrization("2021", "14", [("", 1588)], result=3048, dataset_class=DataSet,
                                  steps=10)
round_2 = dataset_parametrization("2021", "14", [("", 2188189693529)], result=3288891573057,
                                  dataset_class=DataSet, steps=40)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_day_14(dataset: DataSet):
    polymer, mapping = dataset.get_data()
    for _ in range(dataset.params["steps"]):
        polymer = do_step(polymer, mapping)
    result_sorted = list(sorted(count(polymer).values()))
    assert result_sorted[-1] - result_sorted[0] == dataset.result
