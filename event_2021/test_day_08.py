from pathlib import Path
import pytest


number_of_segments = {0: 6, 1: 2, 2: 5, 3: 5, 4: 4, 5: 5, 6: 6, 7: 3, 8: 7, 9: 6}
unique_segments = (2, 3, 4, 7)


def get_data(input_file: str):
    lines = Path(input_file).read_text().splitlines()
    return [[list(map(set, x.split())) for x in line.split('|')] for line in lines]


@pytest.mark.parametrize("input_file", ["input/day_08_example.txt", "input/day_08.txt"])
class TestDay08:
    def test_part_one(self, input_file: str):
        data = get_data(input_file)
        uniques = [w for line in data for w in line[1] if len(w) in unique_segments]
        result = len(uniques)
        print(f"Unique digits appear {result} times.")
        assert result == (26 if "example" in input_file else 445)

    def test_part_two(self, input_file: str):
        data = get_data(input_file)
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
        print(f"The sum of all displays is {result}.")
        assert result == (61229 if "example" in input_file else 1043101)
