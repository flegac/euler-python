from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Tuple, Set, Iterable, List

from euler.lib.graph import Graph, power
from euler.lib.timer import show_timers, timer, reset_timers


class Problem(object):
    def __init__(self, N: int):
        self.A = set(list(range(N)))
        self.N = N

    def canonical_all_ok(self, x: List[int]):
        y = list(reversed(x))
        for i in range(len(x)):
            part = y[:i + 1]
            if self.full_check(part):
                z = list(reversed(part))
                return z
        return x

    def canonical_all_ko(self, x: List[int]):
        y = list(reversed(x))
        tot = 0
        for i in range(len(y)):
            tot += y[i]
            if tot >= self.N:
                z = list(reversed(y[:i + 1]))
                return z
        return x

    def canonical_mixed(self, x: List[int]):
        y = list(reversed(x))
        sign = self._signature(y)
        i = sign.index(True)
        invalid, valid = y[:i], y[i:]

        if sum(invalid) == self.N - 1:
            z = list(reversed(invalid))
            return z
        z = invalid + self.canonical_all_ok(valid)
        z = list(reversed(z))
        return z

    @timer
    def canonical_form(self, x: List[int]):
        if x == [0]:
            return x
        y = list(filter(lambda _: _ != 0, reversed(x)))

        if sum(y) <= self.N:
            return list(reversed(y))

        if self.full_check(x):
            z = self.canonical_all_ok(x)
        else:
            sign = self._signature(y)

            if True in sign:
                z = self.canonical_mixed(x)
            else:
                z = self.canonical_all_ko(x)
        return z

    @timer
    def adjacent(self, x: List[int]):
        res = {from_digits(x)}
        for i in self.A:
            if i == 0:
                continue
            y = x + [i]
            if not self.is_impossible(y):
                # z = self.canonical_form_backup(y)
                z = self.canonical_form(y)
                z = from_digits(z)
                res.add(z)
        return res

    @timer
    def build_graph(self):
        if Path('graph.json').exists():
            return Graph.from_path('graph.json')
        gg = defaultdict(list)
        visited = set()
        to_visit = set(self.A)
        while len(to_visit) > 0:
            x = to_visit.pop()
            visited.add(x)
            xx = digits(x)
            # if sum(xx) > self.N:
            #     print(x)
            for y in self.adjacent(xx):
                gg[x].append(y)
                if y not in visited:
                    to_visit.add(y)

        g = Graph(gg)
        g.save('graph.json')
        return g

    @timer
    def check(self, x: List[int]):
        return sum(x) == self.N

    @timer
    def _signature(self, x: List[int]):
        test = [False] * len(x)
        for i in range(len(x)):
            for j in range(i + 1, len(x) + 1):
                w = x[i:j]
                if self.check(w):
                    for a in range(i, j):
                        test[a] = True
        return test

    @timer
    def full_check(self, x: List[int]):
        x = list(filter(lambda _: _ != 0, x))
        n = len(x)
        last = 0
        for i in range(n):
            cpt = 0
            for j in range(i + 1, n + 1):
                cpt += x[j - 1]
                if cpt == self.N:
                    if i > last:
                        return False
                    last = j
        res = last == n
        return res

    @timer
    def is_impossible(self, x: List[int]):
        for i in self.A:
            w = x + [i]
            if self.full_check(w):
                return False
        return True


def partition(number) -> Set[Tuple[int]]:
    answer = set()
    answer.add((number,))
    for x in range(1, number):
        for y in partition(number - x):
            answer.add(tuple(((x,) + y)))
    answer = sorted(answer)
    answer = sorted(answer, key=lambda x: len(x))
    return answer


@timer
def digits(x: int):
    return [int(_) for _ in str(x)]


@timer
def from_digits(x: Iterable[int]):
    return int(''.join(map(str, x)))


def L(n: int, k: int = 1):
    test = Problem(10).full_check

    A = list(range(k, 10))
    for w in product(*[A] * n):
        if test(w):
            yield w


def p529(n: int):
    '''

    0) Definitions
    - Let A = {0,...,9}
    - Let B = {1,...,9}
    - Let L be the language { u / u is 10-substring friendly }
    - Let L(n) be the language {u in L / u_0 != 0, |u| = n }
    - Let L(n;k) be the set {u in L(n) / |u|_0 = k } # no zero in u

    1) More definitions:
    - Let X be the minimal automata (number of vertices) that generates all L(n;0)
    - Let V(X) be the set of accepting states of X
    - Let M be the adjacency matrix of X
    - Let M(n) be the n-power of M


    1) Prove that:
        |L(n)| = sum(k=0..n) [ |L(n-k;0)| * binomial(n,k) ]
        |L(n;0)| = sum(u in B) sum(v in V(n)) [ M(n)[u][v] ]

     4) Full formulae is :
        |L(n)| = sum(k=0..n) sum(u in B) sum(v in V(n)) [ M(n-k)[u][v] * binomial(n,k) ]

    5) Compute the result
        - pre-computes all M(n)
        - pre-computes all |L(n;0)|



  k=[0, 1, 2, 3, 4, 5]
n=2 [9, 9, 7, 5, 3, 1]
n=3 [72, 45, 22, 8, 3, 1]
n=4 [507, 273, 104, 34, 9, 1]
n=5 [3492, 1587, 422, 83, 15, 1]
n=6 [23697, 9045, 1742, 263, 33, 1]
n=7 [158940, 50979, 7174, 727, 63, 1]

    '''

    problem = Problem(10)

    g = problem.build_graph()

    V = list(filter(lambda x: problem.full_check(digits(x)), g.vertices))
    print('terminal:', V)
    print('vertex:', len(g.vertices))

    mat = g.matrix()

    @timer
    def perf_test(n: int):
        m = mat[:n, :n]
        m2 = m.dot(m)

    for i in range(1, 10):
        perf_test(200 * i)
        print(200*i, '***************')
        show_timers()
        reset_timers()

    M = power(mat, 2, 1_000_000_7)
    print(M)
    # tot = 0
    # for a in problem.A:
    #     for b in V:
    #         i = g.vertices.index(a)
    #         j = g.vertices.index(b)
    #         val = M[i, j]
    #         tot += val
    # print(tot)


if __name__ == '__main__':
    p529(5)

    show_timers()
