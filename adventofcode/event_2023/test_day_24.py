"""
--- Day 24: Never Tell Me The Odds ---
https://adventofcode.com/2023/day/24
"""
import logging
from dataclasses import dataclass
from functools import partial
from itertools import starmap, combinations
from typing import Iterator

import numpy as np
import pytest
import sympy
import tinyarray as ta
from more_itertools import quantify
from sympy import Line, Point, Line3D, Point3D, symbols, solve, diff

from adventofcode.utils import dataset_parametrization, DataSetBase, cross_product

YEAR= "2023"
DAY= "24"

@dataclass(frozen=True)
class HailTrajectory:
    pos: ta.ndarray_int
    v: ta.ndarray_int

    def position_at_time(self, t: float) -> ta.ndarray_float:
        return self.pos + t * self.v

    def two_points(self) -> tuple[ta.ndarray_float, ta.ndarray_float]:
        return self.position_at_time(0) / 10**14, self.position_at_time(10**13) / 10**14

    def time_at_position(self, pos: ta.ndarray_float) -> float | None:
        t = np.linalg.norm(pos - self.pos) / np.linalg.norm(self.v)
        if np.allclose(np.array(self.pos) + t * np.array(self.v), np.array(pos)):
            return t
        if np.allclose(np.array(self.pos) - t * np.array(self.v), np.array(pos)):
            return -t
        return None

    def intersect(self, other: "HailTrajectory") -> int | None:
        t = round(np.linalg.norm(self.pos - other.pos) / np.linalg.norm(other.v - self.v))
        if t * self.v + self.pos == t * other.v + other.pos:
            return t
        if -t * self.v + self.pos == -t * other.v + other.pos:
            return -t
        return None


class DataSet(DataSetBase):
    def trajectories(self) -> Iterator[HailTrajectory]:
        for line in self.lines():
            pos_str, v_str = line.split(sep='@', maxsplit=1)
            yield HailTrajectory(ta.array((*map(int, pos_str.split(',')),)), ta.array((*map(int, v_str.split(',')),)))

    def lines3d(self) -> Iterator[Line3D]:
        """ Round 2 of the problem is solved with sympy, convert HailTrajectories to Line3D"""
        for trajectory in self.trajectories():
            yield Line3D(Point3D(trajectory.position_at_time(0)), Point3D(trajectory.position_at_time(1)))

round_1 = dataset_parametrization(
    year=YEAR, day=DAY, examples=[("", (7, 27, 2))], result=(2*10**14, 4*10**14, 15107), dataset_class=DataSet)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", 47)],
                                  result=856642398547748, dataset_class=DataSet)


def intersect_x_y(h1: HailTrajectory, h2: HailTrajectory) -> np.ndarray | None:
    a = np.array((h1.v, -h2.v)).T
    try:
        inv_a = np.linalg.inv(a[:2])
    except np.linalg.LinAlgError:
        logging.info("Parallel: %s %s", h1, h2)
        return None
    return inv_a@(np.array(h2.pos-h1.pos)[:2])


def intersect_2d_test_area(h1: HailTrajectory, h2: HailTrajectory, minimum: int, maximum: int) -> bool:
    solution = intersect_x_y(h1, h2)
    if solution is not None:
        if np.all(solution >= 0):
            pos = h1.position_at_time(solution[0])
            return minimum <= pos[0] <= maximum and minimum <= pos[1] <= maximum
    return False

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    minimum, maximum, result = dataset.result
    assert quantify(starmap(partial(intersect_2d_test_area, minimum=minimum, maximum=maximum),
                            combinations(dataset.trajectories(), 2))) == result


def intersecting_four_lines(line1: Line3D, line2: Line3D, line3: Line3D, line4: Line3D) -> set[Line3D]:
    """
    Construct the hyperboloid of one sheet which contains the first three lines, then intersect it with the
    fourth line to get two candidate points (one candidate point if the fourth line is a tangent on the hyperboloid).
    For each candidate point, construct the tangent plane on the hyperboloid and intersect it with the hyperboloid.
    This defines two lines, one of which intersects all four lines.

    Return a set of lines (zero, one or two) which intersect all four lines. Note that the edge case of infinitely many
    solutions is not implemented and raises a NotImplementedError.

    The solutions are returned as SymPy symbolic expressions without loss of precision.

    Credits for the hyperboloid equation to https://math.stackexchange.com/a/4221095/1272832
    """
    a1, b1 = line1.points
    a2, b2 = line2.points
    a3, b3 = line3.points
    c1 = b1 - a1
    c2 = b2 - a2
    c3 = b3 - a3
    d1 = cross_product(a1, b1)
    d2 = cross_product(a2, b2)
    d3 = cross_product(a3, b3)
    x, y, z = symbols("x y z", real=True)
    hp = (d1[0] * (c2[1] * c3[2] - c2[2] * c3[1]) \
            + d2[0] * (c3[1] * c1[2] - c3[2] * c1[1]) \
            + d3[0] * (c1[1] * c2[2] - c1[2] * c2[1])) * x**2 \
         + (d1[1] * (c2[2] * c3[0] - c2[0] * c3[2]) \
            + d2[1] * (c3[2] * c1[0] - c3[0] * c1[2]) \
            + d3[1] * (c1[2] * c2[0] - c1[0] * c2[2])) * y**2 \
         + (d1[2] * (c2[0] * c3[1] - c2[1] * c3[0]) \
            + d2[2] * (c3[0] * c1[1] - c3[1] * c1[0]) \
            + d3[2] * (c1[0] * c2[1] - c1[1] * c2[0])) * z**2 \
         + (c1[0] * (c2[1] * d3[1] - d2[1] * c3[1] + c3[2] * d2[2] - d3[2] * c2[2]) \
            + c2[0] * (c3[1] * d1[1] - d3[1] * c1[1] + c1[2] * d3[2] - d1[2] * c3[2]) \
            + c3[0] * (c1[1] * d2[1] - d1[1] * c2[1] + c2[2] * d1[2] - d2[2] * c1[2])) * y * z \
         + (c1[1] * (c2[2] * d3[2] - d2[2] * c3[2] + c3[0] * d2[0] - d3[0] * c2[0]) \
            + c2[1] * (c3[2] * d1[2] - d3[2] * c1[2] + c1[0] * d3[0] - d1[0] * c3[0]) \
            + c3[1] * (c1[2] * d2[2] - d1[2] * c2[2] + c2[0] * d1[0] - d2[0] * c1[0])) * z * x \
         + (c1[2] * (c2[0] * d3[0] - d2[0] * c3[0] + c3[1] * d2[1] - d3[1] * c2[1]) \
            + c2[2] * (c3[0] * d1[0] - d3[0] * c1[0] + c1[1] * d3[1] - d1[1] * c3[1]) \
            + c3[2] * (c1[0] * d2[0] - d1[0] * c2[0] + c2[1] * d1[1] - d2[1] * c1[1])) * x * y \
         + (d1[0] * (c2[1] * d3[1] - d2[1] * c3[1] - c3[2] * d2[2] + d3[2] * c2[2]) \
            + d2[0] * (c3[1] * d1[1] - d3[1] * c1[1] - c1[2] * d3[2] + d1[2] * c3[2]) \
            + d3[0] * (c1[1] * d2[1] - d1[1] * c2[1] - c2[2] * d1[2] + d2[2] * c1[2])) * x \
         + (d1[1] * (c2[2] * d3[2] - d2[2] * c3[2] - c3[0] * d2[0] + d3[0] * c2[0]) \
            + d2[1] * (c3[2] * d1[2] - d3[2] * c1[2] - c1[0] * d3[0] + d1[0] * c3[0]) \
            + d3[1] * (c1[2] * d2[2] - d1[2] * c2[2] - c2[0] * d1[0] + d2[0] * c1[0])) * y \
         + (d1[2] * (c2[0] * d3[0] - d2[0] * c3[0] - c3[1] * d2[1] + d3[1] * c2[1]) \
            + d2[2] * (c3[0] * d1[0] - d3[0] * c1[0] - c1[1] * d3[1] + d1[1] * c3[1]) \
            + d3[2] * (c1[0] * d2[0] - d1[0] * c2[0] - c2[1] * d1[1] + d2[1] * c1[1])) * z \
         + d1[0] * (d2[1] * d3[2] - d2[2] * d3[1]) + d1[1] * (d2[2] * d3[0] - d2[0] * d3[2]) \
         + d1[2] * (d2[0] * d3[1] - d2[1] * d3[0])
    solutions = solve(line4.equation(x, y, z) + (hp,), (x, y, z))
    if solutions[0][0].free_symbols:
        raise NotImplementedError("There is an infinite set of lines intersecting all four input lines.")
    dx = diff(hp, x)
    dy = diff(hp, y)
    dz = diff(hp, z)
    result = set()
    for a in solutions:
        p = dx.subs(dict(zip((x, y, z), a)))*(x-a[0]) \
            + dy.subs(dict(zip((x,y,z), a)))*(y-a[1]) \
            + dz.subs(dict(zip((x,y,z), a)))*(z-a[2])
        candidates = sympy.solve((p, hp), (x, y, z))
        for candidate_line in (Line(Point(c).subs(z, 0), Point(c).subs(z, 1)) for c in candidates):
            if candidate_line.intersect(line1):
                result.add(candidate_line)
    return result


def get_stone_start_point(dataset: DataSet) -> Point3D:
    """ Get four lines from the dataset, so that no two lines are parallel. Get a fifth line as control line.
    Get the set of lines which intersect all the first four lines. Select the solution by choosing the line
    which also intersects the control line. Construct the point at t=0 by inspecting the times at two of the
    intersection points, then follow the solution line to t=0.
    """
    t = symbols("t")
    line_iterator = dataset.lines3d()
    lines = []
    control_line = None
    while len(lines) != 4:
        line = next(line_iterator)
        if any(line.is_parallel(l) for l in lines):
            control_line = line
        else:
            lines.append(line)
    if control_line is None:
        control_line = next(line_iterator)
    # pylint: disable=no-value-for-parameter
    line_candidates = intersecting_four_lines(*lines)
    for line in line_candidates:
        for intersection in line.intersect(control_line):
            a, b = intersection, lines[0].intersection(line)[0]
            t1 = control_line.parameter_value(a, t=t)[t]
            t2 = lines[0].parameter_value(b, t=t)[t]
            v = (b-a)/(t2-t1)
            return a - t1 * v
    # No solution found
    assert False


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert sum(get_stone_start_point(dataset)) == dataset.result
