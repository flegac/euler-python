import json
from collections import defaultdict
from math import log2
from pathlib import Path
from typing import Dict

from scipy.sparse import dok_matrix

from euler.lib.automat import Automat
from euler.lib.graph import Graph
from euler.lib.timer import timer


class EmptyMatrix(object):
    @staticmethod
    def from_automat(g: Automat):
        g0 = {
            k: [_[1] for _ in v]
            for k, v in g.graph.items()
        }

        mat = EmptyMatrix()
        for i in g0:
            for j in g0[i]:
                mat.graph[i][j] += 1

        xx = dict()
        for i in mat.graph:
            if i not in xx:
                xx[i] = list()
            for j in mat.graph[i]:
                for _ in range(mat.graph[i][j]):
                    xx[i].append(j)

        assert xx.keys() == g0.keys()
        for i in xx.keys():

            if sorted(xx[i]) != sorted(g0[i]):
                print(i, xx[i])
                print(i, g0[i])

        assert g.size() == mat.size(), '{} {}'.format(g.size(), mat.size())
        return mat

    @staticmethod
    def from_graph(g: Graph):
        mat = EmptyMatrix()
        for i in g.graph:
            for j in g.graph[i]:
                mat.graph[i][j] += 1
        return mat

    @staticmethod
    def from_path(path: str):
        with open(path) as _:
            graph = json.load(_)
            gg = {
                int(k): {
                    int(k2): v
                    for k2, v in graph[k].items()
                }
                for k in graph
            }
            return EmptyMatrix(gg)

    def save(self, path: str):
        print('saving matrix:', path)
        graph = self._graph()
        with open(path, 'w') as _:
            json.dump(graph, _, indent=4, sort_keys=True)

    def __init__(self, graph: Dict[int, Dict[int, int]] = None):
        self.graph = graph or defaultdict(lambda: defaultdict(lambda: 0))

    def _graph(self):
        res = dict()
        for x in self.graph:
            res[x] = dict(self.graph[x])
        return res

    def transpose(self):
        res = EmptyMatrix()
        for x in self.graph:
            for y in self.graph[x]:
                res.graph[y][x] += self.graph[x][y]
        return res

    def __repr__(self):
        return str(self._graph())

    @staticmethod
    def mult(x: 'EmptyMatrix', y: 'EmptyMatrix'):
        return compute_empty_mult(x, y)

    def size(self):
        return sum(sum(self.graph[x].values()) for x in self.graph)


import numpy as np


def to_sparse(a: EmptyMatrix, vv: Dict[int, int]):
    v = list(sorted(vv.keys()))
    size = len(vv)

    mat = dok_matrix((size, size), dtype=np.uint)
    for x in a.graph:
        for y in a.graph[x]:
            mat[vv[x], vv[y]] = 1
    return mat


@timer
def compute_empty_mult(a: EmptyMatrix, b: EmptyMatrix):
    x = a.graph
    y = b.graph
    res = EmptyMatrix()
    all = len(x)
    progress = all // 5
    for c, i in enumerate(x):
        if c % progress == 0:
            print(c, '/', all)
        for j in y[i]:
            for k in b.graph:
                if k in x[i] and k in y and j in y[k]:
                    prod = x[i][k] * y[k][j]
                    if prod != 0:
                        res.graph[i][j] += prod

    return res


@timer
def power_all(path: str, m: EmptyMatrix, n: int):
    if n == 1:
        return m

    full_path = '{}_{}.json'.format(path, n)
    if Path(full_path).exists():
        return EmptyMatrix.from_path(full_path)

    k = int(log2(n))
    x = power2(path, m, k)
    if 2 ** k == n:
        return x

    y = power_all(path, m, n - 2 ** k)
    res = EmptyMatrix.mult(x, y)
    res.save(full_path)
    return res


@timer
def power2(path: str, m: EmptyMatrix, k: int):
    full_path = '{}_2_{}.json'.format(path, k)
    if Path(full_path).exists():
        return EmptyMatrix.from_path(full_path)

    if k == 0:
        m.save(full_path)
        return m

    m_prev = power2(path, m, k - 1)
    m2 = EmptyMatrix.mult(m_prev, m_prev)
    m2.save(full_path)
    return m2
