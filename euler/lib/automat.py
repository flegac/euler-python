import json
from typing import Dict, Iterable, Tuple

import numpy as np

from euler.lib.timer import timer


class Automat(object):
    @staticmethod
    def from_path(path: str):
        with open(path) as _:
            return Automat(json.load(_))

    def __init__(self, graph: Dict[int, Iterable[Tuple[int, int]]]):
        self.graph = {
            int(k): list(sorted(v, key=lambda _: _[0]))
            for k, v in graph.items()
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
            for _, b in self.graph[a]:
                j = vertices.index(b)
                res[i, j] += 1
        return res

    def size(self):
        return sum(len(self.graph[x]) for x in self.graph)


def minimize_automat(automat: Automat):
    # TODO
    V = automat.vertices
    n = len(V)
    sep = np.zeros((n, n))

    for a in V:
        for b in V:
            if automat.graph[a] != automat.graph[b]:
                sep[a, b] = 1
