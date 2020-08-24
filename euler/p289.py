import itertools
from typing import Tuple, List

PartialGrid = Tuple[Tuple[int], Tuple[int], Tuple[int]]


class Mapping(object):
    def __init__(self, mapping: List[int]):
        self.mapping = mapping
        self.left = None
        self.right = None
        self.top = None
        self.bottom = None

    def link(self, i: int):
        i = self.mapping.index(i)
        j = i + 1 if i % 2 == 0 else i - 1
        return self.mapping[j]


def vlink(a: Mapping, b: Mapping):
    assert a.bottom is None
    assert b.top is None
    a.bottom = b
    b.top = a


def hlink(a: Mapping, b: Mapping):
    assert a.right is None
    assert b.left is None
    a.right = b
    b.left = a


ops = list(map(Mapping, [
    [0, 1, 2, 3, 4, 5, 6, 7],  # A
    [0, 1, 2, 5, 3, 4, 6, 7],  # B
    [0, 5, 1, 2, 3, 4, 6, 7],  # C
    [0, 5, 1, 4, 2, 3, 6, 7],  # D
    [0, 3, 1, 2, 4, 5, 6, 7],  # E
    [0, 1, 2, 3, 4, 7, 5, 6],  # F
    [0, 3, 1, 2, 4, 7, 5, 6],  # G
    [0, 7, 1, 2, 3, 4, 5, 6],  # H
    [0, 7, 1, 4, 2, 3, 5, 6],  # I
    [0, 7, 1, 2, 3, 6, 4, 5],  # J
    [0, 7, 1, 6, 2, 3, 4, 5],  # K
    [0, 7, 1, 6, 2, 5, 3, 4],  # L
    [0, 1, 2, 7, 3, 4, 5, 6],  # M
    [0, 1, 2, 7, 3, 6, 4, 5],  # N
]))

if __name__ == '__main__':
    xx = list(itertools.product(range(14), repeat=6))
    xx = list(map(Mapping, xx))

    print(len(xx))
    print(xx[0].mapping)
