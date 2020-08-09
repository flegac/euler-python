from collections import defaultdict
from typing import List, Iterable

from euler.lib.graph import Graph
from euler.lib.timer import timer


class P529(object):
    def __init__(self, n: int):
        self.A = set(list(range(n)))
        self.N = n

    def item(self, n: int):
        return Digits(self, digits(n))

    def canonical_all_ok(self, x: 'Digits'):
        y = x.mirror()
        for i in range(len(x.digits)):
            part = y.slice(last=i + 1)
            if part.full_check():
                return part.mirror()
        return x

    def canonical_all_ko(self, x: 'Digits'):
        y = x.mirror()
        tot = 0
        for i in range(len(y.digits)):
            tot += y.digits[i]
            if tot >= self.N:
                z = y.slice(last=i + 1).mirror()
                return z
        return x

    def canonical_mixed(self, x: 'Digits'):
        y = x.mirror()
        sign = y.sgn
        i = sign.index(True)
        invalid, valid = y.slice(last=i), y.slice(first=i)

        if sum(invalid.digits) == self.N - 1:
            z = invalid.mirror()
            return z
        z = Digits(self, invalid.digits + self.canonical_all_ok(valid).digits).mirror()
        return z

    @timer
    def canonical_form(self, x: 'Digits'):
        if x.digits == [0]:
            return x
        y = x.clean([0]).mirror()

        if sum(y.digits) <= self.N:
            return y.mirror()

        if x.full_check():
            z = self.canonical_all_ok(x)
        else:
            if True in y.sgn:
                z = self.canonical_mixed(x)
            else:
                z = self.canonical_all_ko(x)
        return z

    @timer
    def build_graph(self):
        # if Path('graph.json').exists():
        #     return Graph.from_path('graph.json')
        gg = defaultdict(list)
        visited = set()
        to_visit = set(map(self.item, self.A))
        while len(to_visit) > 0:
            x = to_visit.pop()
            visited.add(x)
            for y in x.adjacent:
                gg[x].append(y)
                if y not in visited:
                    to_visit.add(y)

        g = Graph({
            k.x: [_.x for _ in v]
            for k, v in gg.items()
        })
        g.save('graph.json')
        return g


@timer
def digits(x: int):
    return [int(_) for _ in str(x)]


@timer
def from_digits(x: Iterable[int]):
    return int(''.join(map(str, x)))


def mirror(x: int):
    return from_digits(reversed(digits(x)))


class Digits(object):
    def __init__(self, problem: P529, x: List[int]):
        self.problem = problem
        self.x = from_digits(x)
        self.digits = x

    def __str__(self):
        return ':'+str(self.x)

    def __lt__(self, other):
        return self.x < other.x

    def __hash__(self):
        return self.x

    def __eq__(self, other):
        return self.x == other.x

    def slice(self, first: int = 0, last: int = None):
        if last:
            return Digits(self.problem, self.digits[:last])
        return Digits(self.problem, self.digits[first:])

    def mirror(self):
        return Digits(self.problem, list(reversed(self.digits)))

    def clean(self, values: List[int]):
        if self.digits == [0]:
            return self

        return Digits(self.problem, list(filter(lambda _: _ not in values, self.digits)))

    @property
    def adjacent(self):
        return self.compute_adjacent()

    @timer
    def compute_adjacent(self):
        res = {self}
        for i in self.problem.A:
            if i == 0:
                continue
            y = Digits(self.problem, self.digits + [i])
            if not y._is_impossible():
                z = self.problem.canonical_form(y)
                res.add(z)
        return res

    @timer
    def _is_impossible(self):
        for i in self.problem.A:
            w = Digits(self.problem, self.digits + [i])
            if w.full_check():
                return False
        return True

    @property
    def sgn(self):
        return self.compute_signature()

    @timer
    def compute_signature(self):
        x = self.digits
        test = [False] * len(x)
        for i in range(len(x)):
            for j in range(i + 1, len(x) + 1):
                w = x[i:j]
                if sum(w) == self.problem.N:
                    for a in range(i, j):
                        test[a] = True
        return test

    @timer
    def full_check(self):
        # TODO: check if this is equivalent ?
        # n = len(self.digits)

        x = self.clean([0])
        n = len(x.digits)

        last = 0
        for i in range(n):
            cpt = 0
            for j in range(i + 1, n + 1):
                cpt += x.digits[j - 1]
                if cpt == self.problem.N:
                    if i > last:
                        return False
                    last = j
        res = last == n
        return res
