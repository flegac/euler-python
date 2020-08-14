from typing import List

from euler.lib.timer import timer
from euler.prob_529.p529 import from_digits


class Digit529(object):
    from euler.prob_529.p529 import P529

    def __init__(self, problem: P529, x: List[int]):
        self.problem = problem
        self.digits = x

    def __repr__(self):
        return str(self.value)

    def __lt__(self, other):
        return self.digits < other.digits

    def __hash__(self):
        return hash(tuple(self.digits))

    def __eq__(self, other):
        return self.digits == other.digits

    @property
    def value(self):
        return from_digits(self.digits)

    def slice(self, first: int = 0, last: int = None):
        if last:
            return Digit529(self.problem, self.digits[:last])
        return Digit529(self.problem, self.digits[first:])

    def mirror(self):
        return Digit529(self.problem, list(reversed(self.digits)))

    def clean(self, values: List[int]):
        if self.digits == [0]:
            return self

        return Digit529(self.problem, list(filter(lambda _: _ not in values, self.digits)))

    @property
    def adjacent(self):
        return self.compute_adjacent()

    @timer
    def compute_adjacent(self):
        from euler.prob_529.canonical_form import canonical_form

        res = {(0, self)}
        for i in self.problem.A:
            if i == 0:
                continue
            y = Digit529(self.problem, self.digits + [i])
            if not y._is_impossible():
                z = canonical_form(y)
                res.add((i, z))
        return res

    @timer
    def _is_impossible(self):
        for i in self.problem.A:
            if i == 0:
                continue
            w = Digit529(self.problem, self.digits + [i])
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
        n = len(self.digits)
        last = 0
        for i in range(n):
            cpt = 0
            for j in range(i + 1, n + 1):
                cpt += self.digits[j - 1]
                if cpt == self.problem.N:
                    if i > last:
                        return False
                    last = j
        res = last == n
        return res
