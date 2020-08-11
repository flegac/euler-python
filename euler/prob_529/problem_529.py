from collections import defaultdict
from pathlib import Path
from typing import Iterable

from euler.lib.automat import Automat
from euler.lib.timer import timer


class P529(object):
    def __init__(self, n: int):
        self.A = set(list(range(n)))
        self.N = n

    def item(self, n: int):
        from euler.prob_529.digit529 import Digit529
        return Digit529(self, digits(n))

    @timer
    def build_graph(self):
        if Path('automat.json').exists():
            return Automat.from_path('automat.json')
        gg = defaultdict(list)
        visited = set()
        to_visit = set(map(self.item, self.A))
        while len(to_visit) > 0:
            x = to_visit.pop()
            visited.add(x)
            for a, y in x.adjacent:
                gg[x].append((a, y))
                if y not in visited:
                    to_visit.add(y)

        g = Automat({
            from_digits(k.digits): [(a, from_digits(b.digits)) for a, b in v]
            for k, v in gg.items()
        })
        g.save('automat.json')
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
