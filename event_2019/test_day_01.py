from pathlib import Path


def calculate_fuel(mass: int, recursive=False):
    fuel = mass // 3 - 2
    if not recursive:
        return fuel
    return 0 if fuel < 0 else fuel + calculate_fuel(fuel, recursive=True)


def test_round_01():
    mass = 0
    for line in Path("input/day_01.txt").read_text().splitlines():
        mass += calculate_fuel(int(line))
    assert mass == 3398090


def test_round_02():
    mass = 0
    for line in Path("input/day_01.txt").read_text().splitlines():
        mass += calculate_fuel(int(line), recursive=True)
    assert mass == 5094261
