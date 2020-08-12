from collections import defaultdict
from typing import Mapping, Dict

import numpy as np
from scipy.sparse import dok_matrix

from euler.lib.fast_power import fast_power
from euler.lib.timer import timer
from euler.prob_529.empty_matrix import EmptyMatrix, power_all
from euler.prob_529.problem_529 import P529

MOD = 1_000_000_007


class P529Solver(object):
    SOURCE_NODES = {0: 1}

    @timer
    def __init__(self):
        problem = P529(10)
        self.graph = problem.build_graph()

        self.vv = {
            v: i
            for i, v in enumerate(self.verts)
        }
        self.mat = EmptyMatrix.from_automat(self.graph)
        self.sparse = to_sparse(self.mat, self.vv)
        print('solver initialized :')
        print('initial:', self.graph.initial)
        print('terminals:', len(self.graph.terminals))
        print('states:', len(self.graph.states))

    @property
    def verts(self):
        return  self.graph.states

    @property
    def terminals(self):
        return self.graph.terminals

    def enumerate(self, n: int):
        xx = P529Solver.SOURCE_NODES
        for i in range(n + 1):
            count = self.word_number(xx)
            yield count
            last = xx
            xx = self.next_nodes(xx)
            if last == xx:
                break

    @timer
    def compute_value2(self, n: int):
        if n == 0:
            return 0
        mat2 = self.mat_power2(n)
        tot = 0
        for v1 in P529Solver.SOURCE_NODES:
            for v2 in self.terminals:
                tot += mat2.graph[v1].get(v2, 0)
        return tot

    @timer
    def next_nodes(self, nodes: Mapping):
        res = defaultdict(lambda: 0)
        for v1 in nodes:
            for _, v2 in self.graph.graph[v1]:
                res[v2] += nodes[v1]
                res[v2] %= MOD
        return res

    @timer
    def word_number(self, nodes: Mapping):
        tot = 0
        for v in self.terminals:
            tot += nodes.get(v, 0)
        tot %= MOD
        return tot

    @timer
    def compute_value(self, n: int):
        if n == 0:
            return 0
        mat2 = self.mat_power(n)
        tot = 0
        for v1 in P529Solver.SOURCE_NODES:
            for v2 in self.terminals:
                tot += mat2[self.vv[v1], self.vv[v2]]
        return tot

    def mat_power(self, n: int):
        def mult(a, b):
            return a.dot(b)

        return fast_power(self.sparse, n, mult)

    def mat_power2(self, n: int):
        workspace = '../resources/p529/mat'
        return power_all(workspace, self.mat, n)


def to_sparse(a: EmptyMatrix, vv: Dict[int, int]):
    v = list(sorted(vv.keys()))
    size = len(vv)

    mat = dok_matrix((size, size), dtype=np.uint)
    for x in a.graph:
        for y in a.graph[x]:
            mat[vv[x], vv[y]] = 1
    return mat
