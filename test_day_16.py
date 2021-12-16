import pytest
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union
from bitarray import bitarray
import bitarray.util as ba_util


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


def get_data(input_file) -> bitarray:
    hex_string = Path(input_file).read_text()
    return ba_util.int2ba(int(hex_string, 16), length=4*len(hex_string))


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
    else:
        return packet.version + sum(sum_version(p) for p in packet.sub_packets)


@pytest.mark.parametrize("input_file,expected",
                         (("input/day_16_example_01.txt", 6),
                          ("input/day_16_example_02.txt", 9),
                          ("input/day_16_example_03.txt", 14),
                          ("input/day_16_example_04.txt", 16),
                          ("input/day_16_example_05.txt", 12),
                          ("input/day_16_example_06.txt", 23),
                          ("input/day_16_example_07.txt", 31),
                          ("input/day_16.txt", 969)))
def test_day_16(input_file, expected):
    bits = get_data(input_file)
    packet, rest = parse_packet(bits)
    assert ba_util.ba2int(rest) == 0
    result = sum_version(packet)
    assert result == expected
