from pathlib import Path


number_of_segments = {0: 6, 1: 2, 2: 5, 3: 5, 4: 4, 5: 5, 6: 6, 7: 3, 8: 7, 9: 6}
unique_segments = (2, 3, 4, 7)


def get_data(input_file: str):
    lines = Path(input_file).read_text().splitlines()
    return [[list(map(set, x.split())) for x in line.split('|')] for line in lines]


def part_one(input_file: str):
    data = get_data(input_file)
    uniques = [w for line in data for w in line[1] if len(w) in unique_segments]
    print(f"Unique digits appear {len(uniques)} times.")


def part_two(input_file: str):
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


if __name__ == '__main__':
    [part_one(input_file) for input_file in ("input/day_08_example.txt", "input/day_08.txt")]
    [part_two(input_file) for input_file in ("input/day_08_example.txt", "input/day_08.txt")]
