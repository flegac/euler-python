from collections import defaultdict
from typing import Mapping

import numpy as np
from scipy.sparse import dok_matrix, csc_matrix
from scipy.sparse.linalg import inv

from euler.lib.fast_power import fast_power
from euler.lib.timer import timer
from euler.prob_529.empty_matrix import EmptyMatrix, to_sparse, power_all, power2
from euler.prob_529.problem_529 import P529


class P529Solver(object):
    def __init__(self):
        problem = P529(10)
        self.graph = problem.build_graph()
        self.verts = list(sorted(self.graph.vertices))
        self.terminal = list(filter(lambda _: problem.item(_).full_check(), self.verts))

        self.vv = {
            v: i
            for i, v in enumerate(self.verts)
        }
        self.mat = EmptyMatrix.from_automat(self.graph)
        self.sparse = to_sparse(self.mat, self.vv)

    def solve(self, n: int):
        size = len(self.verts)
        vv = self.vv

        mat = dok_matrix((size, size), dtype=np.uint)
        for x in self.graph.graph:
            for _, y in self.graph.graph[x]:
                mat[vv[x], vv[y]] += 1
        mat_csc = csc_matrix(mat)
        mat_inv = inv(mat_csc)

        print(mat_inv)

    def initial_nodes(self):
        return {0: 1}

    @timer
    def enumerate(self, n: int):
        count = 0
        xx = self.initial_nodes()
        i = 1
        while i <= n:
            i += 1
            yield count

            xx = self.next_nodes(xx)
            count = self.word_number(xx)

    def next_nodes(self, nodes: Mapping):
        res = defaultdict(lambda: 0)
        for v1, kk in nodes.items():
            for a, v2 in self.graph.graph[v1]:
                res[v2] += kk
        return res

    def word_number(self, nodes: Mapping):
        tot = 0
        for v in self.terminal:
            tot += nodes.get(v, 0)
        return tot

    @timer
    def compute_value(self, n: int):
        if n == 0:
            return 0
        mat2 = self.mat_power(n)
        xx = self.initial_nodes()
        tot = 0
        for v1, kk in xx.items():
            for v2 in self.terminal:
                tot += kk * mat2[self.vv[v1], self.vv[v2]]
        return tot

    def mat_power(self, n: int):
        def mult(a, b):
            return a.dot(b)

        mat2 = fast_power(self.sparse, n, mult)
        return mat2

    def value(self, n: int):
        workspace = '../resources/p529/mat'

        power2(workspace, self.mat, 10)
        mat2 = power_all(workspace, self.mat, n)

        xx = self.initial_nodes()
        xx = {
            i:1
            for i in range(1,10)
        }
        tot = 0
        for v1, kk in xx.items():
            for v2 in self.terminal:
                tot += kk * mat2.graph[v1].get(v2,0)
        return tot
