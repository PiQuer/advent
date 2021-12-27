from pathlib import Path
import numpy as np
import pytest


program = Path("input/day_24.txt").read_text().splitlines()


def char_to_index(a):
    return int.from_bytes(a.encode(), 'big') - 119


def get_value(register, b):
    try:
        return int(b)
    except ValueError:
        return register[char_to_index(b)]


def inp(register, input_iterator, a):
    register[char_to_index(a)] = next(input_iterator)


def add(register, a, b):
    b = get_value(register, b)
    register[char_to_index(a)] += b


def mul(register, a, b):
    b = get_value(register, b)
    register[char_to_index(a)] *= b


def div(register, a, b):
    b = get_value(register, b)
    register[char_to_index(a)] /= b


def mod(register, a, b):
    b = get_value(register, b)
    register[char_to_index(a)] %= b


def eql(register, a, b):
    b = get_value(register, b)
    register[char_to_index(a)] = register[char_to_index(a)] == b


def process(input_data, prg):
    assert len(str(input_data)) == 14
    input_iterator = (int(s) for s in str(input_data))
    register = np.zeros(4, dtype=int)
    linenum = 0
    inputnum = 0
    for line in prg:
        linenum += 1
        instr = line.split()
        if instr[0] == 'inp':
            inp(register, input_iterator, *instr[1:])
            inputnum += 1
        elif instr[0] == 'add':
            add(register, *instr[1:])
        elif instr[0] == 'mul':
            mul(register, *instr[1:])
        elif instr[0] == 'div':
            div(register, *instr[1:])
        elif instr[0] == 'mod':
            mod(register, *instr[1:])
        elif instr[0] == 'eql':
            eql(register, *instr[1:])
    return register[-1]


@pytest.mark.parametrize("input_data,expected",
                         ((91297395919993, 0),
                          (71131151917891, 0)))
def test_day_24(input_data, expected):
    assert process(input_data, program) == expected
