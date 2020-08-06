from collections import defaultdict
from math import ceil, log2, floor

import numpy as np


class Graph(object):
    @staticmethod
    def from_dict(graph: dict):
        res = Graph()
        res.graph = graph
        return res

    def __init__(self):
        self.graph = defaultdict(set)

    def link(self, v1: int, v2: int):
        len(self.graph[v2])
        self.graph[v1].add(v2)

    def matrix(self) -> np.ndarray:
        vertices = list(self.graph.keys())
        n = len(self.graph)
        res = np.zeros((n, n))
        for a in self.graph:
            i = vertices.index(a)
            for b in self.graph[a]:
                j = vertices.index(b)
                res[i, j] = 1
        return res


def power(matrix: np.ndarray, n: int, modulo: int = None):
    if n == 0:
        return matrix
    power2 = [matrix]
    for i in range(ceil(log2(n))):
        m = power2[-1].dot(power2[-1])
        if modulo:
            m %= modulo
        power2.append(m)

    i = 1
    m = matrix
    while i < n:
        p = floor(log2(n - i))
        i += 2 ** p
        m = m.dot(power2[p])
        if modulo:
            m %= modulo

    return m
