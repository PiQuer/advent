import pytest
from pathlib import Path
from collections import defaultdict


def get_data(input_file):
    mapping = {}
    with Path(input_file).open('r') as f:
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


@pytest.mark.parametrize("input_file, steps, expected",
                         (("input/day_14_example.txt", 10, 1588),
                          ("input/day_14.txt", 10, 3048),
                          ("input/day_14_example.txt", 40, 2188189693529),
                          ("input/day_14.txt", 40, 3288891573057)))
def test_day_14(input_file, steps, expected):
    polymer, mapping = get_data(input_file)
    for _ in range(steps):
        polymer = do_step(polymer, mapping)
    result_sorted = list(sorted(count(polymer).values()))
    result = result_sorted[-1] - result_sorted[0]
    assert result == expected
