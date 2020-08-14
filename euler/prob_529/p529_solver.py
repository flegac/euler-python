import json
from collections import defaultdict
from typing import Mapping

from scipy.sparse import csr_matrix

from euler.lib.automate import Automate, sparse_matrix, update_classes, minimize
from euler.lib.fast_power import FastPower
from euler.lib.timer import timer
from euler.prob_529.p529 import P529

MOD = 1_000_000_007


class P529Solver(object):
    SOURCE_NODES = {'0': 1}

    @timer
    def __init__(self, use_mod: bool = False):
        self.use_mod = use_mod
        self.automate = P529(10).build_automate()
        print('solver initialized :')
        self.automate.show_stats()

        with open('../resources/p529/partitions.json') as _:
            parts = json.load(_)
        print('partition:', len(parts))
        parts = list(parts.values())

        self.automate = update_classes(self.automate, parts)
        self.automate.save('automat_small.json')
        self.automate.show_stats()

        xx = minimize(self.automate)

        @timer
        def matrix_mult(a, b):
            x = a.dot(b)
            if use_mod:
                x = x.todense()
                x %= MOD
                x = csr_matrix(x)
            try:
                print('matrix.size:', x.nnz)
            except:
                pass

            return x

        self.matrix = sparse_matrix(self.automate)
        self.mat_power = FastPower(self.matrix, matrix_mult, path='../resources/p529/mat')

    @property
    def terminals(self):
        return self.automate.T

    def enumerate(self, n: int):
        xx = P529Solver.SOURCE_NODES
        for i in range(n + 1):
            count = self.word_number(xx)
            yield count
            xx = self.next_nodes(xx, self.automate)

    @timer
    def next_nodes(self, states: Mapping, automate: Automate):
        res = defaultdict(lambda: 0)
        for v1 in states:
            for _, a, v2, n in automate.transitions(v1):
                res[v2] += states[v1] * n
                if self.use_mod:
                    res[v2] %= MOD
        return dict(res)

    @timer
    def word_number(self, nodes: Mapping):
        tot = 0
        for v in self.terminals:
            tot += nodes.get(v, 0)
            if self.use_mod:
                tot %= MOD
        return tot

    @timer
    def compute_value(self, n: int):
        if n == 0:
            return 0

        mat2 = self.mat_power.power(n)
        tot = 0
        for v1 in P529Solver.SOURCE_NODES:
            for v2 in self.terminals:
                tot += mat2[self.automate.index[v1], self.automate.index[v2]]
                if self.use_mod:
                    tot %= MOD
        return tot
