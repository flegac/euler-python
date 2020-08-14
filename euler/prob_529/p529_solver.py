from collections import defaultdict
from typing import Mapping, Dict

import numpy as np
from scipy.sparse import dok_matrix

from euler.lib.automate import Automate
from euler.lib.fast_power import fast_power
from euler.lib.timer import timer
from euler.prob_529.empty_matrix import EmptyMatrix, power_all
from euler.prob_529.p529 import P529

MOD = 1_000_000_007


class P529Solver(object):
    SOURCE_NODES = {0: 1}

    @timer
    def __init__(self, use_mod: bool = False):
        self.use_mod = use_mod
        self.automate = P529(10).build_automate()

        print('solver initialized :')
        print('states:', len(self.automate.S))
        print('initial:', {self.automate.I})
        print('terminals:', len(self.automate.T))

        # for i, depth in enumerate(reversed(analyze_graph(self.automate))):
        #     print('depth {}:'.format(i), len(depth), list(reversed(depth)))

        # self.mat = EmptyMatrix.from_automat(self.automate)
        # self.sparse = to_sparse(self.mat, self.automate.index)

    @property
    def terminals(self):
        return self.automate.T

    def enumerate(self, n: int):
        xx = P529Solver.SOURCE_NODES

        g2: Automate = None
        x2 = None

        series = defaultdict(list)
        for i in range(n + 1):
            count = self.word_number(xx)
            yield count
            xx = self.next_nodes(xx, self.automate)
            class_number, class_sizes, classes = self.compute_stats(xx)

            if g2:
                x2 = self.next_nodes(x2, g2)
                count2 = self.word_number(x2)
                print('------------------------------------------')
                print(count)
                print(count2)
                # stats = {
                #     c[0]: xx[c[0]]
                #     for c in classes
                # }
                # for c in stats:
                #     series[c].append(stats[c])

            # if class_number == 2816 and not g2:
            #     x2 = {
            #         int(c[0]): xx[c[0]] * len(c)
            #         for c in classes
            #     }
            #
            #     g2 = P529(10).build_graph('automat.json')
            #     g2.update_classes(classes)
            #     g2.save('automat2.json')

        # with open('series.txt', 'w') as _:
        #     for c in sorted(series, key=lambda x: series[x][0]):
        #         _.write('{:20} {}\n'.format(c, series[c]))

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
    def compute_value2(self, n: int):
        if n == 0:
            return 0
        mat2 = self.mat_power2(n)
        tot = 0
        for v1 in P529Solver.SOURCE_NODES:
            for v2 in self.terminals:
                tot += mat2.automate[v1].get(v2, 0)
        return tot

    @timer
    def next_nodes(self, nodes: Mapping, automate: Automate):
        res = defaultdict(lambda: 0)
        for v1 in nodes:
            for _, a, v2, n in automate.transitions(str(v1)):
                res[v2] += nodes[v1] * n
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
