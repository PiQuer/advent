import pytest
from pathlib import Path
from typing import Optional

Interval = tuple[int, int]


class Cuboid:
    def __init__(self, xmin: int, xmax: int, ymin: int, ymax: int, zmin: int, zmax: int):
        if xmax <= xmin or ymax <= ymin or zmax <= zmin:
            self._borders = ((0, 0), (0, 0), (0, 0))
        else:
            self._borders = ((xmin, xmax), (ymin, ymax), (zmin, zmax))

    def __str__(self):
        return str(self._borders)

    def __repr__(self):
        return str(self)

    def __eq__(self, other: "Cuboid"):
        return self._borders == other._borders

    def __hash__(self):
        return hash(self._borders)

    @staticmethod
    def _i_div(a, b, other_a, other_b) -> tuple[Interval, Interval, Interval]:
        return (a, min(other_a, b)), (max(other_b, a), b), (max(a, other_a), min(b, other_b))

    def __sub__(self, other: "Cuboid") -> "PolyCuboid":
        result = set()
        reduced_borders = ()
        for dim in (0, 1, 2):
            before, after, reduced = self._i_div(*self._borders[dim], *other._borders[dim])
            reduced_borders += (reduced,)
            for exc in (before, after):
                result.add(Cuboid(*sum(reduced_borders[:dim], ()), *exc, *sum(self._borders[dim+1:], ())))
        return PolyCuboid(result)

    def __and__(self, other: "Cuboid") -> "Cuboid":
        i = {}
        for dim in (0, 1, 2):
            i[dim] = (max(self._borders[dim][0], other._borders[dim][0]),
                      min(self._borders[dim][1], other._borders[dim][1]))
        return Cuboid(*i[0], *i[1], *i[2])

    def __or__(self, other: "Cuboid") -> "PolyCuboid":
        result1 = {self} | (other - self).cuboid_set
        result2 = {other} | (self - other).cuboid_set
        return PolyCuboid(result1) if len(result1) <= len(result2) else PolyCuboid(result2)

    def __abs__(self) -> int:
        result = 1
        for dim in range(3):
            result *= self._borders[dim][1] - self._borders[dim][0]
        return result


empty = Cuboid(0, 0, 0, 0, 0, 0)


class PolyCuboid:
    def __init__(self, cuboids: Optional[set[Cuboid]] = None):
        self._cuboids: set[Cuboid] = cuboids if cuboids is not None else set()
        self._cuboids.add(empty)

    @property
    def cuboid_set(self) -> set[Cuboid]:
        return self._cuboids

    def __str__(self):
        return '{ ' + '\n  '.join(str(c) for c in self._cuboids) + ' }'

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._cuboids)

    def __eq__(self, other: "PolyCuboid"):
        return self._cuboids == other._cuboids

    def __hash__(self):
        return hash(self._cuboids)

    def __sub__(self, other: Cuboid):
        result = set()
        for c in self._cuboids:
            result |= (c - other)._cuboids
        return PolyCuboid(result)

    def __or__(self, other: Cuboid):
        return PolyCuboid({other} | (self - other)._cuboids)

    def __and__(self, other: Cuboid):
        result = set()
        for c in self._cuboids:
            result |= {c & other}
        return PolyCuboid(result)

    def __abs__(self) -> int:
        return sum(abs(c) for c in self._cuboids)


def get_data(input_file):
    for line in Path(input_file).open("r"):
        on_off, coords_str = line.split()
        coords = []
        for c in coords_str.split(','):
            i, j = map(int, c[2:].split('..'))
            coords.extend((i, j+1))
        yield on_off, Cuboid(*coords)


@pytest.mark.parametrize("input_file,expected_part_one,expected_part_two",
                         (("input/day_22_example_01.txt", 39, 39),
                          ("input/day_22_example_02.txt", 590784, 39769202357779),
                          ("input/day_22_example_03.txt", 474140, 2758514936282235),
                          ("input/day_22.txt", 568000, 1177411289280259)))
def test_part_one(input_file, expected_part_one, expected_part_two):
    pc = PolyCuboid()
    mask = Cuboid(-50, 51, -50, 51, -50, 51)
    for on_off, cuboid in get_data(input_file):
        if on_off == 'on':
            pc |= cuboid
        else:
            pc -= cuboid
    result_part_one = abs(pc & mask)
    result_part_two = abs(pc)
    assert result_part_one == expected_part_one
    assert result_part_two == expected_part_two
