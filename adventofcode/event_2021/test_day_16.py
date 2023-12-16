"""
--- Day 16: Packet Decoder ---
https://adventofcode.com/2021/day/16
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

import bitarray.util as ba_util
import numpy as np
from bitarray import bitarray

from adventofcode.utils import dataset_parametrization, DataSetBase, generate_rounds


class TypeID(Enum):
    OPERATOR_SUM = 0
    OPERATOR_PROD = 1
    OPERATOR_MIN = 2
    OPERATOR_MAX = 3
    LITERAL = 4
    OPERATOR_GT = 5
    OPERATOR_LT = 6
    OPERATOR_EQ = 7


@dataclass
class Packet:
    version: int
    type_id: TypeID


@dataclass
class OperatorPacket(Packet):
    sub_packets: List[Packet]


@dataclass
class LiteralPacket(Packet):
    content: int


def get_packet(input_file) -> Packet:
    hex_string = Path(input_file).read_text(encoding="ascii")
    bits = ba_util.int2ba(int(hex_string, 16), length=4*len(hex_string))
    packet, rest = parse_packet(bits)
    assert len(rest) == 0 or ba_util.ba2int(rest) == 0
    return packet


def parse_literal(bits: bitarray) -> Tuple[int, bitarray]:
    result = bitarray()
    count = 0
    while True:
        more = bits[0]
        count += 1
        result.extend(bits[1:5])
        bits = bits[5:]
        if not more:
            return ba_util.ba2int(result), bits


def parse_operator(bits: bitarray) -> Tuple[List[Packet], bitarray]:
    length_id, bits = bits[0], bits[1:]
    result = []
    if length_id == 0:
        num_bits, bits = ba_util.ba2int(bits[:15]), bits[15:]
        bits, rest = bits[:num_bits], bits[num_bits:]
        while True:
            if len(bits) == 0:
                return result, rest
            packet, bits = parse_packet(bits)
            result.append(packet)
    if length_id == 1:
        num_packets, bits = ba_util.ba2int(bits[:11]), bits[11:]
        for _ in range(num_packets):
            packet, bits = parse_packet(bits)
            result.append(packet)
        return result, bits
    assert False


def parse_packet(bits: bitarray) -> Tuple[Packet, bitarray]:
    version, type_id, rest = ba_util.ba2int(bits[0:3]), TypeID(ba_util.ba2int(bits[3:6])), bits[6:]
    if type_id == TypeID.LITERAL:
        content, rest = parse_literal(rest)
        packet = LiteralPacket(version=version, type_id=type_id, content=content)
    else:
        sub_packets, rest = parse_operator(rest)
        packet = OperatorPacket(version=version, type_id=type_id, sub_packets=sub_packets)
    return packet, rest


def sum_version(packet: Union[Packet, LiteralPacket, OperatorPacket]):
    if packet.type_id == TypeID.LITERAL:
        return packet.version
    return packet.version + sum(sum_version(p) for p in packet.sub_packets)


def process(packet: Union[Packet, LiteralPacket, OperatorPacket]):
    if packet.type_id == TypeID.LITERAL:
        return packet.content
    sub = list(process(p) for p in packet.sub_packets)
    match packet.type_id:
        case TypeID.OPERATOR_SUM:
            result = np.sum(sub)
        case TypeID.OPERATOR_PROD:
            result = np.prod(sub)
        case TypeID.OPERATOR_MIN:
            result = np.min(sub)
        case TypeID.OPERATOR_MAX:
            result = np.max(sub)
        case TypeID.OPERATOR_GT:
            result = int(sub[0] > sub[1])
        case TypeID.OPERATOR_LT:
            result = int(sub[0] < sub[1])
        case TypeID.OPERATOR_EQ:
            result = int(sub[0] == sub[1])
        case _:
            assert False
    return result

round_1 = dataset_parametrization(
    "2021", "16", [("_01", 6), ("_02", 9), ("_03", 14), ("_04", 16), ("_05", 12), ("_06", 23), ("_07", 31)],
    result=969, fn=sum_version)
round_2 = dataset_parametrization(
    "2021", "16", [("_08", 3), ("_09", 54), ("_10", 7), ("_11", 9), ("_12", 1), ("_13", 0), ("_14", 0), ("_15", 1)],
    result=124921618408, fn=process)
pytest_generate_tests = generate_rounds(round_1, round_2)


def test_day_16(dataset: DataSetBase):
    packet = get_packet(dataset.input_file)
    assert dataset.params["fn"](packet) == dataset.result
