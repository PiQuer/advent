import pytest
import copy
from typing import Union, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SnailNumber:
    left: Union[int, "SnailNumber"]
    right: Union[int, "SnailNumber"]

    @classmethod
    def _from_str(cls, number_str, i) -> tuple[int, Union[int, "SnailNumber"]]:
        if number_str[i] == '[':
            i, left = cls._from_str(number_str, i+1)
            assert number_str[i] == ','
            i, right = cls._from_str(number_str, i+1)
            assert number_str[i] == ']'
            i += 1
            return i, SnailNumber(left=left, right=right)
        n = i
        while number_str[i].isdecimal():
            i += 1
        return i, int(number_str[n:i])

    @classmethod
    def from_str(cls, number_str: str) -> "SnailNumber":
        _, result = cls._from_str(number_str, 0)
        return result

    def __str__(self):
        result = "["
        if isinstance(self.left, int):
            result += str(self.left)
        else:
            result += str(self.left)
        result += ","
        if isinstance(self.right, int):
            result += str(self.right)
        else:
            result += str(self.right)
        result += "]"
        return result

    def __repr__(self):
        return self.__str__()

    def _add_helper(self, n, which_part):
        other = "left" if which_part == "right" else "right"
        element = self.__getattribute__(other)
        if isinstance(element, int):
            self.__setattr__(other, element + n)
            return
        sn = self.__getattribute__(other)
        while isinstance(sn.__getattribute__(which_part), SnailNumber):
            sn = sn.__getattribute__(which_part)
        sn.__setattr__(which_part, sn.__getattribute__(which_part) + n)

    def _explode(self, level=1) -> tuple[Optional[int], Optional[int], bool]:
        if level > 4:
            return self.left, self.right, True
        result = self._explode_helper(level, which_part="left")
        if result[2]:
            return result
        result = self._explode_helper(level, which_part="right")
        return result

    def _explode_helper(self, level, which_part: str):
        select = 0 if which_part == "left" else 1
        select_other = 1 - select
        element = (self.left, self.right)[select]

        def add_fn(x):
            if x:
                self._add_helper(x, which_part=which_part)
        left = right = None
        explode_happened = False
        if isinstance(element, SnailNumber):
            left, right, explode_happened = element._explode(level + 1)
            if level == 4:
                self.__setattr__(which_part, 0)
        add_fn((left, right)[select_other])
        return (left, None)[select], (right, None)[select_other], explode_happened

    def _split(self) -> bool:
        if isinstance(self.left, int):
            if self.left >= 10:
                self.left = SnailNumber(left=self.left >> 1, right=(self.left >> 1) + (self.left & 1))
                return True
        else:
            if self.left._split():
                return True
        if isinstance(self.right, int):
            if self.right >= 10:
                self.right = SnailNumber(left=self.right >> 1, right=(self.right >> 1) + (self.right & 1))
                return True
            return False
        return self.right._split()

    def reduce(self) -> None:
        more = True
        while more:
            _, _, exploded = self._explode()
            more = exploded or self._split()

    def __add__(self, other):
        result = SnailNumber(left=copy.deepcopy(self), right=copy.deepcopy(other))
        result.reduce()
        return result

    def __abs__(self):
        return 3 * (self.left if isinstance(self.left, int) else abs(self.left)) \
            + 2 * (self.right if isinstance(self.right, int) else abs(self.right))


def get_data(input_file):
    return [SnailNumber.from_str(line) for line in Path(input_file).read_text().splitlines()]


@pytest.mark.parametrize("number,exploded",
                         (("[[[[[9,8],1],2],3],4]", "[[[[0,9],2],3],4]"),
                          ("[7,[6,[5,[4,[3,2]]]]]", "[7,[6,[5,[7,0]]]]"),
                          ("[[3,[2,[1,[7,3]]]],[6,[5,[4,[3,2]]]]]", "[[3,[2,[8,0]]],[9,[5,[4,[3,2]]]]]"),
                          ("[[3,[2,[8,0]]],[9,[5,[4,[3,2]]]]]", "[[3,[2,[8,0]]],[9,[5,[7,0]]]]")))
def test_explode(number, exploded):
    number = SnailNumber.from_str(number)
    exploded = SnailNumber.from_str(exploded)
    _, _, explode_happened = number._explode()
    assert explode_happened
    assert number == exploded


@pytest.mark.parametrize("number,split",
                         (("[10,1]", "[[5,5],1]"),
                          ("[1,10]", "[1,[5,5]]"),
                          ("[11,5]", "[[5,6],5]"),
                          ("[5,11]", "[5,[5,6]]"),
                          ("[1,[15,[10,10]]]", "[1,[[7,8],[10,10]]]")))
def test_split(number, split):
    number = SnailNumber.from_str(number)
    split = SnailNumber.from_str(split)
    split_happened = number._split()
    assert split_happened
    assert number == split


@pytest.mark.parametrize("number, reduced",
                         (("[[[[[4,3],4],4],[7,[[8,4],9]]],[1,1]]", "[[[[0,7],4],[[7,8],[6,0]]],[8,1]]"),))
def test_reduce(number, reduced):
    number = SnailNumber.from_str(number)
    reduced = SnailNumber.from_str(reduced)
    number.reduce()
    assert number == reduced


@pytest.mark.parametrize("input_file, expected",
                         (("input/day_18_example.txt", 4140),
                          ("input/day_18.txt", 4435)))
def test_part_one(input_file, expected):
    numbers = get_data(input_file)
    result = abs(sum(numbers[1:], start=numbers[0]))
    assert result == expected
