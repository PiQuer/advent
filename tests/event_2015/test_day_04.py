"""
--- Day 4: The Ideal Stocking Stuffer ---
https://adventofcode.com/2015/day/4
"""
from hashlib import md5
from itertools import count

from utils import dataset_parametrization, DataSetBase, generate_rounds

round_1 = dataset_parametrization("2015", "04", examples=[("1", 609043), ("2", 1048970)], result=282749,
                                  prefix='0'*5)
round_2 = dataset_parametrization("2015", "04", examples=[], result=9962624, prefix='0'*6)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_round_1(dataset: DataSetBase):
    pretext = dataset.text()
    prefix = dataset.params["prefix"]
    assert next(i for i in count() if md5(f"{pretext}{i}".encode()).hexdigest().startswith(prefix)) \
           == dataset.result
