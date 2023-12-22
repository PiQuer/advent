"""
--- Day 20: Pulse Propagation ---
https://adventofcode.com/2023/day/20
"""
import abc
import math
import operator
import warnings
from collections import deque
from dataclasses import dataclass
from functools import partial

import pytest
import tinyarray as ta
from more_itertools import only

from adventofcode.utils import dataset_parametrization, DataSetBase

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
        self.state = False

    def process(self, pulse: Pulse) -> list[Pulse]:
        if pulse.high:
            return []
        self.state = not self.state
        return self.generate_pulses(self.state)

class Conjunction(Module):
    def __init__(self, name: str, destinations: tuple[str, ...]):
        super().__init__(name=name, destinations=destinations)
        self.inputs = {}

    def register_input(self, source: str):
        self.inputs[source] = False

    def process(self, pulse: Pulse) -> list[Pulse]:
        self.inputs[pulse.source] = pulse.high
        if all(i for i in self.inputs.values()):
            return self.generate_pulses(False)
        return self.generate_pulses(True)


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
round_2 = dataset_parametrization(year=YEAR, day=DAY, examples=[], result=241823802412393, dataset_class=DataSet)


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
    target = 1000
    modules = dataset.modules()
    pulse_count = []
    for _ in range(target):
        pulse_count.append(push_the_button(modules))
    assert operator.mul(*sum(pulse_count)) == dataset.result


def create_graph(modules: dict[str, Module], pulse: Pulse):
    try:
        # pylint: disable=import-outside-toplevel
        from pyvis.network import Network
    except ImportError:
        warnings.warn("pyvis not installed")
        return
    net = Network(notebook=True, directed=True, height="800px")
    net.add_node(n_id="rx", label="rx", color = "#C1C1C1", shape="circle")
    for m in modules.values():
        match str(m)[0]:
            case "C":
                m: Conjunction
                color = "#FF9494"if all(i for i in m.inputs) else "#FF1B1B"
            case "F":
                m: FlipFlop
                color = "#2164FF"if m.state else "#94B4FF"
            case _:
                color = "#C1C1C1"
        net.add_node(n_id=m.name, label=m.name, color=color, shape="circle")
    for m in modules.values():
        for d in m.destinations:
            if m.name == pulse.source and d == pulse.destination:
                value = 2
                title = "high" if pulse.high else "low"
                net.add_edge(source=m.name, to=d, value=value, title=title, arrowStrikethrough=False)
            else:
                net.add_edge(source=m.name, to=d)
    net.show("2023_day_20.html")


def get_prime(b_str: str, modules: dict[str, Module]):
    b = modules[b_str]
    result = ""
    while b is not None:
        result = f"1{result}" if any(isinstance(modules[d], Conjunction) for d in b.destinations) else f"0{result}"
        b = only(m for d in b.destinations if isinstance(m := modules[d], FlipFlop))
    return int(result, 2)


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSet):
    modules = dataset.modules()
    assert math.lcm(*map(partial(get_prime, modules=modules), modules['broadcaster'].destinations)) == dataset.result
