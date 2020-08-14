from pathlib import Path
from typing import Iterable, Union

from euler.lib.automate import Automate
from euler.lib.timer import timer


class P529(object):
    def __init__(self, n: int):
        self.A = set(list(range(n)))
        self.N = n

    def item(self, n: Union[str, int]):
        from euler.prob_529.digit529 import Digit529
        return Digit529(self, digits(int(n)))

    @timer
    def build_automate(self, filename='automat.json'):
        if Path(filename).exists():
            return Automate.from_path(filename)

        g = Automate.from_scratch()

        to_visit = {self.item(0)}
        while len(to_visit) > 0:
            s1 = to_visit.pop()
            for a, s2 in s1.adjacent:
                if str(s2) not in g.S:
                    to_visit.add(s2)
                g.add(str(s1), str(a), str(s2))

        g.I = '0'
        g.compute_terminals(lambda x: self.item(x).full_check())
        g.save(filename)
        return g


@timer
def digits(x: int):
    return [int(_) for _ in str(x)]


@timer
def from_digits(x: Iterable[int]):
    return int(''.join(map(str, x)))


@timer
def mirror(x: int):
    return from_digits(reversed(digits(x)))
