import json
from collections import defaultdict
from typing import Dict, Set, Tuple, Generator, Any, Callable

import numpy as np

from euler.lib.timer import timer
from euler.lib.utils import error, warning

Edge = Tuple[str, str, str, int]


class Automate(object):

    def show_stats(self):
        print('states:', len(self.S))
        print('initial:', {self.I})
        print('terminals:', len(self.T))

        # for i, depth in enumerate(reversed(analyze_graph(self))):
        #     print('depth {}:'.format(i), len(depth), list(reversed(depth)))

    @staticmethod
    def builder():
        return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

    @staticmethod
    def builder_clean(Q: Dict[str, Dict[str, Dict[str, int]]], to_str: Callable[[Any], str] = str):
        return {
            to_str(s1): {
                to_str(a): {
                    to_str(s2): n
                    for s2, n in l2.items()
                }
                for a, l2 in l1.items()
            }
            for s1, l1 in Q.items()
        }

    @staticmethod
    def from_scratch():
        return Automate()

    @staticmethod
    def from_json(data: Dict):
        res = Automate()
        res.A = set(data['A'])
        res.S = set(data['S'])
        res.I = data['I']
        res.T = set(data['T'])
        res.Q = data['Q']
        return res.validate()

    @staticmethod
    def from_path(path: str):
        with open(path) as _:
            data = json.load(_)
        return Automate.from_json(data)

    def __init__(self):
        self.A: Set[str] = set()
        self.S: Set[str] = set()
        self.I: str = '---none---'
        self.T: Set[str] = set()
        self.Q: Dict[str, Dict[str, Dict[str, int]]] = dict()
        self.finalized = False
        self.index = None

    def find(self, s1: str, a: str):
        return self.Q.get(s1, dict()).get(a, dict())

    def compute_terminals(self, is_terminal: Callable[[str], bool]):
        self.T = set(filter(is_terminal, self.S))

    def validate(self):
        missing_transitions = []
        probabilistic_transitions = []
        error(self.I in self.S, 'Initial state must be in S : {}'.format(self.I))

        for x in self.T:
            error(x in self.S, 'Terminal states must be in S : {}'.format(x))
        for s1, letter_edges in self.Q.items():
            if set(letter_edges.keys()) != self.A:
                if len(missing_transitions) < 5:
                    missing_transitions.append(s1)
                elif len(missing_transitions) == 5:
                    missing_transitions.append('...')
            for a, edges in letter_edges.items():
                error(a in self.A, 'Transition must be in A: {}'.format((s1, a)))
                if len(edges) > 1:
                    if len(probabilistic_transitions) < 5:
                        probabilistic_transitions.append((s1, a))
                    elif len(missing_transitions) == 5:
                        missing_transitions.append('...')

                for s2, n in edges.items():
                    error(s2 in self.S, 's2 states is not in S : (s1:{}, a:{}, s2:{}, n:{})'.format(s1, a, s2, n))
                    error(n > 0, 'Transition must be in A')

        # warning(len(missing_transitions) == 0,
        #         'Missing transitions: {}'.format(missing_transitions))
        # warning(len(probabilistic_transitions) == 0,
        #         'Automat is probabilistic: {}'.format(probabilistic_transitions))

        self.index = {
            v: i
            for i, v in enumerate(sorted(self.S))
        }
        self.finalized = True
        return self

    def save(self, path: str):
        self.validate()
        with open(path, 'w') as _:
            json.dump(self.to_json(), _, indent=4, sort_keys=True)

    def to_json(self):
        print()
        return {
            'A': list(sorted(self.A)),
            'S': list(sorted(self.S)),
            'I': self.I,
            'T': list(sorted(self.T)),
            'Q': self.Q
        }

    def size(self):
        return sum(len(self.Q[s]) for s in self.Q)

    def add(self, s1: str, a: str, s2: str, n: int = 1):
        self.A.add(a)
        self.S.add(s1)
        self.S.add(s2)

        if s1 not in self.Q:
            self.Q[s1] = dict()
        if a not in self.Q[s1]:
            self.Q[s1][a] = dict()
        if s2 not in self.Q[s1][a]:
            self.Q[s1][a][s2] = 0
        self.Q[s1][a][s2] += n
        return self

    def transitions(self, s1: str) -> Generator[Edge, Any, None]:
        for a, edges in self.Q[s1].items():
            for s2, n in edges.items():
                edge = s1, a, s2, n
                yield edge

    def edges(self) -> Generator[Edge, Any, None]:
        for s1, letter_edges in self.Q.items():
            for edge in self.transitions(s1):
                yield edge

    @timer
    def to_matrix(self) -> np.ndarray:
        n = len(self.S)
        mat = np.zeros((n, n), dtype=np.uint64)

        for v1, _, v2, n in self.edges():
            i = self.index[v1]
            j = self.index[v2]
            mat[i, j] += n
        return mat


def analyze_graph(automate: Automate):
    states = automate.S.difference(automate.T)
    depths = [automate.T]
    visited = set(automate.T)

    while len(states):
        next_depth = set()
        for s1 in states:
            adjacents = set(s2 for _, a, s2, n in automate.transitions(s1))
            if adjacents.issubset(visited.union([s1])):
                next_depth.add(s1)
        states.difference_update(next_depth)
        depths.append(next_depth)
        visited.update(next_depth)

    return list(map(list, map(sorted, depths)))


@timer
def show_automate(aut: Automate, path: str, N: int = 500):
    from graphviz import Digraph
    dot = Digraph(comment='p529')

    S = set(list(sorted(aut.S))[:N])

    for x in S:
        dot.node(x, x)

    for s1, a, s2, n in aut.edges():
        if s1 == s2:
            continue
        if s1 in S and s2 in S:
            dot.edge(s1, s2)
    dot.render('{}_{}.gv'.format(path, N), view=True)
