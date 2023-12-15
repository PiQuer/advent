"""
--- Day 24: Arithmetic Logic Unit ---
https://adventofcode.com/2021/day/24
"""
import numpy as np

from utils import dataset_parametrization, DataSetBase, generate_rounds


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


round_1 = dataset_parametrization("2021", "24", [], result=0, input_data=91297395919993)
round_2 = dataset_parametrization("2021", "24", [], result=0, input_data=71131151917891)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_day_24(dataset: DataSetBase):
    assert process(dataset.params["input_data"], dataset.lines()) == dataset.result
