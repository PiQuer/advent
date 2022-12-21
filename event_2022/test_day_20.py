"""
https://adventofcode.com/2022/day/20
"""
from functools import partial
from itertools import islice
from operator import attrgetter, mul

import pytest
from dataclasses import dataclass
from typing import Iterator, Optional
from more_itertools import iterate, nth_or_last, consume

from utils import dataset_parametrization, DataSetBase


round_1 = dataset_parametrization(day="20", examples=[("", 3)], result=9945)
round_2 = dataset_parametrization(day="20", examples=[("", 1623178306)], result=3338877775442)


@dataclass
class Node:
    value: int
    id: Optional[int] = None
    previous: Optional["Node"] = None
    next: Optional["Node"] = None

    def __repr__(self):
        return f"Node(value={self.value}, id={self.id}, previous={self.previous.id}, next={self.next.id})"


class LinkedList:
    def __init__(self, node_it: Iterator[Node]):
        self._nodes: dict[int, Node] = dict()
        self._id_zero = None
        n = None
        for node_id, n in enumerate(node_it):
            n.id = node_id
            self._nodes[n.id] = n
            if n.value == 0:
                self._id_zero = node_id
            if node_id:
                n.previous = self._nodes[node_id - 1]
                self._nodes[node_id - 1].next = n
        if n is not None:
            n.next = self._nodes[0]
            self._nodes[0].previous = n

    def mix(self, node_id: int):
        node = self._nodes[node_id]
        target_node = self._target_node(node, node.value)
        if target_node == node.previous:
            return
        node.previous.next = node.next
        node.next.previous = node.previous
        target_node.next.previous = node
        node.next = target_node.next
        node.previous = target_node
        target_node.next = node
        pass

    def _target_node(self, node, value, index_mode=False):
        positions = value % (len(self._nodes) - 1 + int(index_mode))
        if positions == 0:
            return node if index_mode else node.previous
        if positions > (len(self._nodes) - 1 + int(index_mode)) // 2:
            positions -= len(self._nodes) - 1 + int(index_mode)
        target_node = nth_or_last(iterate(attrgetter("next" if positions >= 0 else "previous"), node),
                                  positions if (positions >= 0) else (-positions + 1 - int(index_mode)))
        return target_node

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, item: int):
        return self._target_node(node=self._nodes[self._id_zero], value=item, index_mode=True)

    def __iter__(self):
        return islice(iterate(attrgetter("next"), self._nodes[self._id_zero]), len(self))

    def __reversed__(self):
        return islice(iterate(attrgetter("previous"), self._nodes[self._id_zero].previous), len(self))


@pytest.mark.parametrize(**round_1)
def test_round_1(dataset: DataSetBase):
    linked_list = LinkedList(map(Node, map(int, dataset.lines())))
    consume(map(linked_list.mix, range(len(linked_list))))
    assert sum(map(lambda x: linked_list[x * 1000].value, (1, 2, 3))) == dataset.result


@pytest.mark.parametrize(**round_2)
def test_round_2(dataset: DataSetBase):
    linked_list = LinkedList(map(Node, map(partial(mul, 811589153), map(int, dataset.lines()))))
    for _ in range(10):
        consume(map(linked_list.mix, range(len(linked_list))))
    assert sum(map(lambda x: linked_list[x * 1000].value, (1, 2, 3))) == dataset.result
