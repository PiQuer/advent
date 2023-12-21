"""
--- Day 20: Pulse Propagation ---
https://adventofcode.com/2023/day/20
"""
import abc
import operator
from collections import deque
from dataclasses import dataclass

import pytest
import tinyarray as ta

from adventofcode.utils import dataset_parametrization, DataSetBase

# from adventofcode.utils import generate_rounds

YEAR= "2023"
DAY= "20"

@dataclass(frozen=True)
class Pulse:
    destination: str
    high: bool = False
    source: str = ""


@dataclass
class Module(metaclass=abc.ABCMeta):
    name: str
    destinations: tuple[str, ...]

    @abc.abstractmethod
    def process(self, pulse: Pulse) -> list[Pulse]:
        pass

    def generate_pulses(self, high: bool) -> list[Pulse]:
        return [Pulse(source=self.name, destination=d, high=high) for d in self.destinations]

    def __hash__(self):
        return hash((self.name, self.destinations))

class Broadcaster(Module):
    def __init__(self, destinations: tuple[str, ...]):
        super().__init__(name="broadcaster", destinations=destinations)

    def process(self, pulse: Pulse) -> list[Pulse]:
        return self.generate_pulses(pulse.high)

class Button(Module):
    def __init__(self):
        super().__init__(name="button", destinations=("broadcaster",))

    def process(self, pulse: Pulse) -> list[Pulse]:
        return self.generate_pulses(False)


class FlipFlop(Module):
    def __init__(self, name: str, destinations: tuple[str, ...]):
        super().__init__(name=name, destinations=destinations)
        self._state = False

    def process(self, pulse: Pulse) -> list[Pulse]:
        if not pulse.high:
            self._state = not self._state
            return self.generate_pulses(self._state)
        return []

    def __hash__(self):
        return hash((super().__hash__(), self._state))


class Conjunction(Module):
    def __init__(self, name: str, destinations: tuple[str, ...]):
        super().__init__(name=name, destinations=destinations)
        self._inputs = {}

    def register_input(self, source: str):
        self._inputs[source] = False

    def process(self, pulse: Pulse) -> list[Pulse]:
        self._inputs[pulse.source] = pulse.high
        if all(i for i in self._inputs.values()):
            return self.generate_pulses(False)
        return self.generate_pulses(True)

    def __hash__(self):
        return hash((super().__hash__(), tuple(self._inputs.values())))

class DataSet(DataSetBase):
    def modules(self) -> dict[str, Module]:
        result = {"button": Button()}
        for line in self.lines():
            module_name, _, destination_string = line.split(' ', maxsplit=2)
            destinations = tuple(destination_string.split(', '))
            if module_name == "broadcaster":
                result[module_name] = Broadcaster(destinations=destinations)
            else:
                cls = FlipFlop if module_name[0] == "%" else Conjunction
                result[module_name[1:]] = cls(name=module_name[1:], destinations=destinations)
        for m in result.values():
            for d in m.destinations:
                if isinstance(result.get(d), Conjunction):
                    result[d].register_input(m.name)
        return result

round_1 = dataset_parametrization(year=YEAR, day=DAY, examples=[("1", 32000000), ("2", 11687500)], result=873301506,
                                  dataset_class=DataSet)
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[("", None)], result=None, dataset_class=DataSet)
# pytest_generate_tests = generate_rounds(round_1, round_2)


def push_the_button(modules: dict[str, Module]) -> ta.ndarray_int:
    pulses = deque((Pulse(destination="button"),))
    pulse_count = [0, 0]
    while pulses:
        pulse = pulses.popleft()
        new_pulses = modules[pulse.destination].process(pulse) if pulse.destination in modules else []
        for new_pulse in new_pulses:
            pulse_count[new_pulse.high] += 1
        pulses.extend(new_pulses)
    return ta.array(pulse_count)

@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSet):
    modules = dataset.modules()
    pulse_count = []
    cache = {}
    for n in range(1000):
        key = tuple(modules.values())
        if key in cache:
            break
        cache[key] = n
        pulse_count.append(push_the_button(modules))
    if n != 999:
        first_index = cache[key]
        cycle_length = n - first_index
        cycles, rest = divmod(1000 - first_index, cycle_length)
        result = sum(pulse_count[:first_index]) + sum(pulse_count[first_index:]) * cycles + \
                 sum(pulse_count[first_index:first_index+rest])
    else:
        result = sum(pulse_count)
    assert operator.mul(*result) == dataset.result

@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    assert dataset.result is None
