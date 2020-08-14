from collections import defaultdict
from typing import Mapping

from euler.lib.automate import Automate, sparse_matrix, minimize
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

        xx = minimize(self.automate)


        print('solver initialized :')
        self.automate.show_stats()

        def mult(a, b):
            return a.dot(b)

        self.matrix = sparse_matrix(self.automate)

        self.mat_power = FastPower(self.matrix, mult, path='../resources/p529/mat')

    @property
    def terminals(self):
        return self.automate.T

    def enumerate(self, n: int):
        xx = P529Solver.SOURCE_NODES

        for i in range(n + 1):
            count = self.word_number(xx)
            yield count
            xx = self.next_nodes(xx, self.automate)

            class_number, class_sizes, classes = self.compute_stats(xx)
            if class_number == 2816:
                print(classes)
            print('states:', len(xx), 'class_number:', class_number)

    def compute_stats(self, xx):
        stats = defaultdict(list)
        for k, v in xx.items():
            stats[v].append(k)
        stats = dict(stats)
        class_number = len(stats)
        classes = list(sorted(stats.values(), key=lambda x: len(x), reverse=True))
        class_sizes = list(map(len, classes))
        return class_number, class_sizes, classes

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
        print('matrix.size:', mat2.nnz)
        tot = 0
        for v1 in P529Solver.SOURCE_NODES:
            for v2 in self.terminals:
                tot += mat2[self.automate.index[v1], self.automate.index[v2]]
        return tot
