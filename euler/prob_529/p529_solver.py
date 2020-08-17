import json
from collections import defaultdict
from typing import Mapping

import numpy as np

from euler.lib.automate import Automate
from euler.lib.automate_minimize import update_classes
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
        # print('solver initialized :')
        # self.automate.show_stats()

        # partition = minimize(self.automate)

        with open('../resources/p529/partitions.json') as _:
            parts = json.load(_)
        # print('partition:', len(parts))
        parts = list(parts.values())

        self.automate = update_classes(self.automate, parts)
        self.automate.save('automat_small.json')
        # self.automate.show_stats()

        self.matrix = self.automate.to_matrix()

        # show_automate(self.automate,path='../resources/p529/graphics/', N=100)

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
        tot = np.array([0]).astype(object)
        for v in self.terminals:
            tot += nodes.get(v, 0)
            if self.use_mod:
                tot %= MOD
        return tot[0]

    @timer
    def compute_value(self, n: int, cache_path: str = '../resources/p529/mat'):

        if self.use_mod:
            cache_path += 'mod'

        if n == 0:
            return 0

        @timer
        def flint_matmult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            from flint import nmod_mat, fmpz_mat
            if self.use_mod:
                a = nmod_mat(a.tolist(), MOD)
                b = nmod_mat(b.tolist(), MOD)
            else:
                a = fmpz_mat(a.tolist())
                b = fmpz_mat(b.tolist())

            y = a * b
            z = [list(map(int, _.strip('][').split(','))) for _ in str(y).split('\n')]
            res = np.array(z, dtype=np.uint64)
            return res

        @timer
        def matrix_mult(a, b):
            # use dtype=object for unlimited precision
            a = a.astype(object)
            b = b.astype(object)

            if self.use_mod:
                a %= MOD
                b %= MOD
            # Much faster : but bug with some big integer (even with modulo)
            # a = csr_matrix(a.astype(np.uint64))
            # b = csr_matrix(b.astype(np.uint64))

            x = a.dot(b)
            # x = x.todense()
            if self.use_mod:
                x = x.astype(object)
                x %= MOD
                x = x.astype(np.uint64)
            return x

        mat2 = FastPower(self.matrix, flint_matmult, path=cache_path).power(n)

        tot = np.array([0]).astype(object)
        for v1 in P529Solver.SOURCE_NODES:
            for v2 in self.terminals:
                tot += mat2[self.automate.index[v1], self.automate.index[v2]]
                if self.use_mod:
                    tot %= MOD
        return tot[0]
