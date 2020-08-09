from collections import defaultdict
from typing import List, Iterable

from euler.lib.graph import Graph
from euler.lib.timer import timer


class P529(object):
    def __init__(self, n: int):
        self.A = set(list(range(n)))
        self.N = n

    def item(self, n: int):
        return Digits(self, n)

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
        # if Path('graph.json').exists():
        #     return Graph.from_path('graph.json')
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


@timer
def digits(x: int):
    return [int(_) for _ in str(x)]


@timer
def from_digits(x: Iterable[int]):
    return int(''.join(map(str, x)))


def mirror(x: int):
    return from_digits(reversed(digits(x)))


class Digits(object):
    def __init__(self, problem: P529, x: int):
        self.problem = problem
        self.x = digits(x)
        self.sgn = problem._signature(self.x)
