from typing import Optional, Union
import pytest
from pathlib import Path
import itertools


energies = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
connections = dict()
connections["M0"] = (("M1", 1),)
connections["M1"] = (("M0", 1), ("M2", 2), ("A1", 2))
connections["M2"] = (("M1", 2), ("M3", 2), ("A1", 2), ("B1", 2))
connections["M3"] = (("M2", 2), ("M4", 2), ("B1", 2), ("C1", 2))
connections["M4"] = (("M3", 2), ("M5", 2), ("C1", 2), ("D1", 2))
connections["M5"] = (("M4", 2), ("M6", 1), ("D1", 2))
connections["M6"] = (("M5", 1),)
connections["A1"] = (("M1", 2), ("M2", 2), ("A0", 1))
connections["A0"] = (("A1", 1),)
connections["B1"] = (("M2", 2), ("M3", 2), ("B0", 1))
connections["B0"] = (("B1", 1),)
connections["C1"] = (("M3", 2), ("M4", 2), ("C0", 1))
connections["C0"] = (("C1", 1),)
connections["D1"] = (("M4", 2), ("M5", 2), ("D0", 1))
connections["D0"] = (("D1", 1),)


class State:
    def __init__(self, occupied: Union[dict[str, str], tuple[tuple[str, str]]], moving_out: Optional[str] = None,
                 moving_in: Optional[str] = None):
        if isinstance(occupied, dict):
            # noinspection PyTypeChecker
            self.occupied = tuple(sorted(occupied.items()))
        else:
            self.occupied = occupied
        self.moving_out = moving_out
        self.moving_in = moving_in

    def __str__(self):
        return str(self.occupied)

    def __eq__(self, other: "State"):
        return self.occupied, self.moving_in, self.moving_out == other.occupied, other.moving_in, other.moving_out

    def __hash__(self):
        return hash(self.occupied + (self.moving_in, self.moving_out))


def get_starting_state(input_file) -> State:
    data = Path(input_file).read_text().split()
    line1 = data[2].split('#')[3:7]
    line2 = data[3].split('#')[1:5]
    occupied = {}
    for node, s in zip(("A1", "B1", "C1", "D1"), line1):
        occupied[node] = s
    for node, s in zip(("A0", "B0", "C0", "D0"), line2):
        occupied[node] = s
    return State(occupied)


def move(occupied: dict[str, str], move_from: str, move_to: tuple[str, int]) -> tuple[dict[str, str], int]:
    next_occupied = occupied.copy()
    amphipod = next_occupied[move_from]
    next_occupied[move_to[0]] = amphipod
    del(next_occupied[move_from])
    return next_occupied, energies[amphipod] * move_to[1]


def is_winning(state: State):
    return state.occupied == (('A0', 'A'), ('A1', 'A'), ('B0', 'B'), ('B1', 'B'),
                              ('C0', 'C'), ('C1', 'C'), ('D0', 'D'), ('D1', 'D'))


def get_next_states(state: State, previous: dict[State, tuple[int, Optional[State]]],
                    abort_energy: Optional[int] = None):
    occupied = dict(state.occupied)
    candidate_moves = []
    candidate_states = []
    if state.moving_out is not None and not state.moving_out[0] == 'M':
        candidate_moves.extend((state.moving_out, c) for c in connections[state.moving_out] if c[0][0] == 'M')
    elif state.moving_in is not None and not state.moving_in[0] == 'M':
        candidate_moves.extend((state.moving_in, c) for c in connections[state.moving_in]
                               if c[0][0] == occupied[state.moving_in])
    elif state.moving_in is not None:
        candidate_moves.extend((state.moving_in, c) for c in connections[state.moving_in]
                               if c[0].startswith('M') or c[0].startswith(occupied[state.moving_in]))
    else:
        for key, value in occupied.items():
            candidate_moves.extend(itertools.product((key,), connections[key]))
    for c in candidate_moves:
        if c[1][0] in occupied:
            continue
        amphipod = occupied[c[0]]
        if c[0] == amphipod + "0" or (c[0] == amphipod + "1" and occupied.get(amphipod+"0") == amphipod):
            continue
        moving_in, moving_out = None, None
        if state.moving_in:
            if not c[1][0].startswith('M') and c[1][0][0] != amphipod:
                continue
            if c[1][0] == amphipod + "0" or (c[1][0] == amphipod + "1" and occupied.get(amphipod + "0") == amphipod):
                moving_in = None
        elif state.moving_out:
            if state.moving_out == c[0]:
                if c[0][0] == "M" and c[1][0][0] not in ('M', amphipod):
                    continue
        if c[0].startswith('M'):
            if state.moving_out == c[0]:
                if not c[1][0] == amphipod + "1" and occupied.get(amphipod + "0") == amphipod:
                    moving_out = c[1][0]
            else:
                if occupied.get(amphipod+"0") not in (None, amphipod) \
                        or occupied.get(amphipod+"1") is not None:
                    continue
                moving_in = c[1][0]
        else:
            if c[0][0] != amphipod or (amphipod+"0" in occupied and occupied[amphipod+"0"] != amphipod):
                moving_out = c[1][0]
            elif c[1][0][1] != "0" or amphipod + "0" not in occupied:
                moving_in = c[1][0]
        new_occupied, energy = move(occupied, *c)
        new_state = State(occupied=new_occupied, moving_out=moving_out, moving_in=moving_in)
        new_energy = previous[state][0] + energy
        if new_state in previous:
            if new_energy >= previous[new_state][0]:
                continue
        else:
            candidate_states.append(new_state)
            previous[new_state] = (new_energy, state)
    return candidate_states


@pytest.mark.parametrize("input_file,expected",
                         (("input/day_23_example.txt", 12521),
                          ("input/day_23.txt", 0)))
def test_day_23(input_file, expected):
    start = get_starting_state(input_file)
    previous = {start: (0, None)}
    candidates = [start]
    minimal_winning_energy = None
    counter = 0
    while candidates:
        next_states = []
        for s in candidates:
            if is_winning(s):
                winning_energy = previous[s][0]
                minimal_winning_energy = winning_energy if minimal_winning_energy is None \
                    else min(winning_energy, minimal_winning_energy)
            else:
                next_states.extend(get_next_states(s, previous, minimal_winning_energy))
        candidates = next_states
        counter += 1
    pass
