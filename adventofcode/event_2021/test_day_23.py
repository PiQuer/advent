"""
--- Day 23: Amphipod ---
https://adventofcode.com/2021/day/23
"""
import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property, cache, partial
from itertools import chain, takewhile, islice
from operator import attrgetter, itemgetter, eq
from pathlib import Path
from typing import Optional, Iterator, Any

import numpy as np
import tinyarray as ta
from more_itertools import quantify, first, iterate, always_reversible, rstrip, lstrip
from toolz import complement, count

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_rounds

home_columns = {1: 3, 2: 5, 3: 7, 4: 9}
hw_penalties = (1, 10, 100, 1000, 100, 10, 1)


@dataclass(frozen=True)
class State:
    hallway: tuple[int, ...]
    side_rooms: tuple[tuple[Any, ...]]
    depth: int
    energy: int
    previous: Optional["State"] = None

    @cached_property
    def hw_penalty(self) -> int:
        return ta.dot(self.hallway, hw_penalties)

    @cached_property
    def count_home(self) -> int:
        return self.depth * 4 - quantify(self.hallway) - self.count_move_twice

    @cached_property
    def count_move_twice(self):
        return sum(map(quantify, self.side_rooms))

    def __hash__(self):
        return hash((self.hallway, self.side_rooms))

    def __eq__(self, other: "State"):
        return (self.hallway, self.side_rooms) == (other.hallway, other.side_rooms)

    def __lt__(self, other: "State"):
        return (-self.count_home, self.count_move_twice, self.hw_penalty, self.energy) < \
            (-other.count_home, other.count_move_twice, other.hw_penalty, other.energy)


class DataSet(DataSetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.params['depth'] == 2:
            self.debug = self.debug_positions(Path(__file__).parent / "input" / "day_23_debug_01.txt")
        if self.params['depth'] == 4:
            self.debug = self.debug_positions(Path(__file__).parent / "input" / "day_23_debug_02.txt")
        self.count = 0

    def debug_positions(self, input_file: Path) -> list[State]:
        result = []
        it = iter(input_file.read_text().splitlines())
        lines = list(islice(it, self.params['depth'] + 3))
        while lines:
            result.append(self.get_state(lines))
            lines = list(islice(it, self.params['depth'] + 3))
        return result

    def get_state(self, lines: list[str]) -> State:
        depth = self.params["depth"]
        a = np.array(list(map(list, map(lambda x: x.replace('.', '@').ljust(13).encode(), lines))), dtype=int)
        if len(a) == 5 and depth == 4:
            a = np.concatenate((a, a[-2:]))
            a[-3] = a[-5]
            a[3:5] = np.array(list(map(list, (b"  #D#C#B#A#  ", b"  #D#B#A#C#  "))), dtype=int)
        a -= ord('@')
        hallway = tuple(a[1, (1, 2, 4, 6, 8, 10, 11)])
        side_rooms = tuple(tuple(rstrip(a[2:2+depth, home_columns[x]], partial(eq, x))) for x in range(1, 5))
        return State(hallway=hallway, side_rooms=side_rooms, depth=depth, energy=0)

    def initial_state(self) -> State:
        return self.get_state(self.lines())


def free_hallway(state: State, h_idx: int, s_idx: int, to_side_room=True) -> bool:
    rightwards = h_idx <= s_idx + 1
    step = 1 if rightwards else -1
    s = slice(h_idx + (step if to_side_room else 0), s_idx + (2 if rightwards else 1), step)
    return not any(state.hallway[s])


@cache
def steps(h_idx: int, s_idx: int) -> int:
    return abs(2 * (h_idx - s_idx) - 3) - int(h_idx == 0 or h_idx == 6)


def next_states_moving_in(state: State, seen: dict[State], h_idx: Optional[int] = None,
                          previous: Optional[State] = None) -> Iterator[State]:
    previous = state if previous is None else previous
    loop = enumerate(state.hallway) if h_idx is None else ((h_idx, state.hallway[h_idx]),)
    for h_idx, a in loop:
        s_idx = a - 1
        if a and not any(state.side_rooms[s_idx]) and free_hallway(state, h_idx, s_idx):
            new_energy = state.energy + (steps(h_idx, s_idx) + len(state.side_rooms[s_idx])) * 10**s_idx
            # noinspection PyTypeChecker
            new_state = State(
                hallway=state.hallway[:h_idx] + (0,) + state.hallway[h_idx+1:],
                side_rooms=state.side_rooms[:s_idx] + (state.side_rooms[s_idx][:-1],) + state.side_rooms[a:],
                energy=new_energy, depth=state.depth, previous=previous)
            if new_state not in seen or seen[new_state] > new_state.energy:
                seen[new_state] = new_state.energy
                yield new_state


def next_states_moving_out(state: State, seen: dict[State]) -> Iterator[State]:
    for s_idx, side_room in enumerate(state.side_rooms):
        pos, a = first(lstrip(enumerate(side_room), complement(itemgetter(1))), default=(0, 0))
        if a:
            try_first = a + int(s_idx >= a)
            for h_idx in chain((try_first,), range(try_first), range(try_first + 1, 7)):
                if free_hallway(state, h_idx, s_idx, to_side_room=False):
                    new_energy = state.energy + (steps(h_idx, s_idx) + pos + 1) * 10**(a-1)
                    new_side_room = state.side_rooms[s_idx][:pos] + (0,) + state.side_rooms[s_idx][pos+1:]
                    # noinspection PyTypeChecker
                    new_state = State(
                        hallway=state.hallway[:h_idx] + (a,) + state.hallway[h_idx+1:],
                        side_rooms=state.side_rooms[:s_idx] + (new_side_room,) + state.side_rooms[s_idx+1:],
                        energy=new_energy, depth=state.depth, previous=state)
                    if h_idx == try_first:
                        if (moving_in := first(next_states_moving_in(new_state, seen, h_idx, state), None)) is not None:
                            yield moving_in
                            continue
                    if new_state not in seen or seen[new_state] > new_state.energy:
                        if not (is_blocked_1(new_state, h_idx) or is_blocked_2(new_state)):
                            seen[new_state] = new_state.energy
                            yield new_state
                        else:
                            seen[new_state] = 0


def lower_bound(state: State) -> int:
    seen = defaultdict(lambda: 0)
    result = 0
    for h_idx, a in enumerate(state.hallway):
        if a:
            s_idx = a - 1
            result += (steps(h_idx, s_idx) + 1 + seen[a]) * 10**s_idx
            seen[a] += 1
    for s_idx, side_room in enumerate(state.side_rooms):
        for pos, a in enumerate(side_room):
            if a:
                result += ((pos + 1) + 2 * abs(s_idx - a + 1) + int(s_idx == a - 1) * 2 + (1 + seen[a])) * 10**(a-1)
                seen[a] += 1
    return result


def is_blocked_1(state: State, new_idx) -> bool:
    a1 = state.hallway[new_idx]
    for h_idx, a2 in islice(enumerate(state.hallway), 2, 5):
        if a2 and h_idx != new_idx and (min(a1, a2) < min(h_idx, new_idx) < max(h_idx, new_idx) <= max(a1, a2)):
            return True
    return False


@cache
def count_available_hallway_spots(h_idx: int, hallway: tuple[int, ...]) -> int:
    a = hallway[h_idx]
    s = range(a, -1, -1) if a < h_idx else range(a + 1, 7)

    def _not_blocked(idx: int):
        if not (b := hallway[idx]):
            return True
        return not ((a == b) or (b < h_idx < idx) or (idx < h_idx <= b))
    return count(takewhile(_not_blocked, s))


def is_blocked_2(state: State) -> bool:
    for h_idx in range(2, 5):
        if a := state.hallway[h_idx]:
            s_idx = a - 1
            z = quantify(state.side_rooms[s_idx], lambda b: b and (b == a or a < h_idx <= b or b < h_idx <= a))
            if z > count_available_hallway_spots(h_idx, state.hallway):
                return True
    return False


def visualize(state: State, print_str: bool = False) -> str:
    board = b"#############\n#...........#\n###.#.#.#.###" + b"\n  #.#.#.#.#  " * (state.depth - 1) + \
        b"\n  #########  "
    board = np.array(list(map(list, board.splitlines())), dtype=np.int8)
    for col, side_room in enumerate(state.side_rooms):
        board[2:2+len(side_room), home_columns[col+1]] = ta.array(side_room) + ord('@')
        board[2+len(side_room):2+state.depth, home_columns[col+1]] = col + 1 + ord('@')
    board[1, (1, 2, 4, 6, 8, 10, 11)] = ta.array(state.hallway) + ord('@')
    board = board.view(dtype='S1')
    board[np.where(board == b'@')] = b'.'
    result = b'\n'.join(b''.join(line) for line in board).decode()
    if print_str:
        print(result)
    return result


def evaluate_next(candidates: list[State], seen: dict[State, int], minimum: Optional[State],
                  dataset: DataSet) -> State:
    c = heapq.heappop(candidates)
    dataset.count += 1
    if c.count_home == c.depth * 4:
        if minimum is None or c.energy < minimum.energy:
            return c
    for next_candidate in chain(next_states_moving_in(c, seen), next_states_moving_out(c, seen)):
        if minimum is not None and next_candidate.energy + lower_bound(next_candidate) >= minimum.energy:
            seen[next_candidate] = 0
            continue
        heapq.heappush(candidates, next_candidate)
    return minimum


def play(dataset: DataSet):
    initial = dataset.initial_state()
    candidates = [initial]
    seen = {initial: 0}
    minimum = None
    while candidates:
        minimum = evaluate_next(candidates, seen, minimum, dataset)
    logging.info("Evaluated %d candidates.", dataset.count)
    logging.info("The best path is:\n%s",
                 '\n'.join(always_reversible(
                     map(visualize, takewhile(bool, iterate(attrgetter('previous'), minimum))))))
    return minimum


round_1 = dataset_parametrization(year="2021", day="23",
                                  examples=[("", 12521)], result=12530, dataset_class=DataSet, depth=2)
round_2 = dataset_parametrization(year="2021", day="23",
                                  examples=[("", 44169)], result=50492, dataset_class=DataSet, depth=4)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_day_23(dataset: DataSet):
    assert play(dataset).energy == dataset.result
