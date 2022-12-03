from pathlib import Path
from itertools import cycle


def test_round_1():
    input_data = Path("input/day_01.txt").read_text().splitlines()
    result = 0
    for line in input_data:
        c = int(line[1:])
        result += c if line[0] == "+" else -c
    assert result == 490


def test_round_2():
    input_data = Path("input/day_01.txt").read_text().splitlines()
    result = 0
    seen = {result}
    for line in cycle(input_data):
        c = int(line[1:])
        result += c if line[0] == "+" else -c
        if result in seen:
            break
        seen.add(result)
    assert result == 70357
