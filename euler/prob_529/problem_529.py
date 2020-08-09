from collections import defaultdict
from typing import Iterable

from euler.lib.graph import Graph
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
            from_digits(k.digits): [from_digits(_.digits) for _ in v]
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


@timer
def mirror(x: int):
    return from_digits(reversed(digits(x)))
