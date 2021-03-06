import numpy as np
import pytest
import re
from pathlib import Path
import itertools
from scipy.spatial.transform import Rotation
import math

overlap_threshold = 12

z_rotations = Rotation.from_euler('z', [0, 90, 180, 270], degrees=True)
y_rotations = [(Rotation.from_euler('y', d, degrees=True)*z_rotations).as_matrix().astype(int) for d in (90, -90)]
x_rotations = [(Rotation.from_euler('x', d, degrees=True)*z_rotations).as_matrix().astype(int) for d in (90, 180, -90)]
rotations = np.concatenate(
    (z_rotations.as_matrix().astype(int), np.concatenate(y_rotations), np.concatenate(x_rotations)))


def get_scanner_reports(input_file: str):
    data = re.split(r"\n*--- scanner \d+ ---\n*", Path(input_file).read_text(), flags=re.MULTILINE)[1:]
    return [np.genfromtxt(scanner.splitlines(), delimiter=',', dtype=int) for scanner in data]


def get_distances(scanner_report):
    combinations = list(itertools.combinations(range(scanner_report.shape[0]), r=2))
    return np.concatenate((np.array(combinations), np.array([np.sum(np.diff(scanner_report[combination, :], axis=0)**2)
                                                             for combination in combinations]).reshape(-1, 1)), axis=1)


def get_matching_beacon_candidates(distances_0, distances_1):
    equal = np.nonzero(distances_0[:, 2, None] == distances_1[:, 2])
    return np.concatenate((distances_0[equal[0], :2], distances_1[equal[1], :2]), axis=1)


def align(reports, distances, index_0, index_1):
    candidates = get_matching_beacon_candidates(distances[index_0], distances[index_1])
    if len(candidates) >= math.comb(overlap_threshold, 2):
        for candidate in candidates:
            x1, x2, y1, y2, r = reports[index_0][candidate[0]], reports[index_0][candidate[1]], \
                reports[index_1][candidate[2]], reports[index_1][candidate[3]], None
            for (y1, y2), r in itertools.product(((y1, y2), (y2, y1)), rotations):
                if np.array_equal(np.dot(r, (y2 - y1)), x2 - x1):
                    break
            transformed = np.dot(r, (reports[index_1] - y1).transpose()).transpose() + x1
            # noinspection PyUnresolvedReferences
            if (reports[index_0][:, None] == transformed).all(-1).any(-1).sum() >= overlap_threshold:
                scanner_pos = np.dot(r, -y1) + x1
                reports[index_1][:] = transformed
                return True, scanner_pos
    return False, None


@pytest.mark.parametrize("input_file,expected_one, expected_two",
                         (("input/day_19_example.txt", 79, 3621),
                          ("input/day_19.txt", 462, 12158)))
def test_part_one(input_file, expected_one, expected_two):
    reports = get_scanner_reports(input_file)
    distances = [get_distances(report) for report in reports]
    aligned = [(0, np.array([0, 0, 0], dtype=int))]
    unaligned = list(range(1, len(reports)))
    while unaligned:
        for a, u in itertools.product(aligned, unaligned):
            is_aligned, scanner_pos = align(reports, distances, a[0], u)
            if is_aligned:
                aligned.append((u, scanner_pos))
                unaligned.remove(u)
                break
    result_one = np.unique(np.concatenate(reports), axis=0).shape[0]
    result_two = max(np.sum(np.abs(p[0][1]-p[1][1])) for p in itertools.combinations(aligned, r=2))
    assert result_one == expected_one
    assert result_two == expected_two
