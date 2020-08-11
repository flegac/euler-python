import json
from math import ceil, log2, floor
from typing import Dict, Iterable

import numpy as np

from euler.lib.timer import timer


class Graph(object):
    @staticmethod
    def from_path(path: str):
        with open(path) as _:
            return Graph(json.load(_))

    def __init__(self, graph: Dict[int, Iterable[int]]):
        self.graph = {
            int(k): list(sorted(graph[k]))
            for k in graph
        }
        self.vertices = list(self.graph.keys())

    def save(self, path: str):
        with open(path, 'w') as _:
            json.dump(self.graph, _, indent=4, sort_keys=True)

    @timer
    def matrix(self) -> np.ndarray:
        vertices = self.vertices
        n = len(self.graph)
        res = np.zeros((n, n)).astype(np.uint)
        for a in self.graph:
            i = vertices.index(a)
            for b in self.graph[a]:
                j = vertices.index(b)
                res[i, j] += 1
        return res


@timer
def power(M: np.ndarray, n: int, modulo: int = None):
    M = M
    if n == 0:
        return M
    power2 = [M]  # power2[k] = M^(2^k)
    k = 1
    while 2 ** k <= n:
        m = power2[k - 1]
        m2 = m.dot(m)
        power2.append(m2)
        k += 1

    for i in range(ceil(log2(n))):
        m = power2[-1].dot(power2[-1])
        if modulo:
            m %= modulo
        power2.append(m)

    i = 1
    m = M
    while i < n:
        p = floor(log2(n - i))
        i += 2 ** p
        m = m.dot(power2[p])
        if modulo:
            m %= modulo

    return m
