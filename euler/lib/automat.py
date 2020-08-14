import json
from typing import Dict, Iterable, Tuple, List

import numpy as np
from automata.fa.dfa import DFA

from euler.lib.timer import timer


class Automat(object):
    @staticmethod
    def from_path(path: str):
        with open(path) as _:
            return Automat(json.load(_))

    def __init__(self, graph: Dict[int, Iterable[Tuple[int, int]]]):
        self.graph = {
            int(k): list(sorted(v, key=lambda _: _[0]))
            for k, v in graph.items()
        }
        self.states = list(sorted(self.graph.keys()))
        self.terminals = None
        self.initial = 0

    def save(self, path: str):
        with open(path, 'w') as _:
            json.dump(self.graph, _, indent=4, sort_keys=True)

    def size(self):
        return sum(len(self.graph[x]) for x in self.graph)

    @timer
    def update_classes(self, classes: List[List[int]]):
        # init mapping
        mapping = dict()
        cc = set()
        for c in classes:
            cc.add(c[0])
            for x in c[1:]:
                mapping[x] = c[0]

        for s in mapping:
            del self.graph[s]

        for s, l in self.graph.items():
            for e in l:
                e[1] = mapping.get(e[1], e[1])

        self.states = list(sorted(self.graph.keys()))


def analyze_graph(automat: Automat):
    states = set(automat.states).difference(set(automat.terminals))
    depths = [set(automat.terminals)]
    visited = set(depths[-1])

    while len(states):
        next_depth = set()
        for x in states:
            adjacents = set(y for _, y in automat.graph[x])
            if adjacents.issubset(visited.union([x])):
                next_depth.add(x)
        states.difference_update(next_depth)
        depths.append(next_depth)
        visited.update(next_depth)

    return list(map(list, map(sorted, depths)))


@timer
def show_graph(graph: Automat, N: int = 100):
    from graphviz import Digraph
    dot = Digraph(comment='p529')

    for x in graph.states[:N]:
        dot.node(str(x), str(x))
    for x in graph.states[:N]:
        for a, y in graph.graph[x]:
            if y in graph.states[:N]:
                dot.edge(str(x), str(y))
    dot.render('test-output/p259_{}.gv'.format(N), view=True)


@timer
def matrix(automat: Automat) -> np.ndarray:
    states = automat.states
    n = len(automat.graph)
    res = np.zeros((n, n)).astype(np.uint)
    for a in automat.graph:
        i = states.index(a)
        for _, b in automat.graph[a]:
            j = states.index(b)
            res[i, j] += 1
    return res


def minimize(automat: Automat):
    symbols = set(map(str, list(range(10))))

    def fullfill(l):
        xx = {
            str(a): 'black_hole'
            for a in symbols
        }
        for a, v in l:
            xx[str(a)] = str(v)
        return xx

    transitions = {
        str(k): fullfill(l)
        for k, l in automat.graph.items()
    }
    transitions['black_hole'] = fullfill([])

    aa = DFA(
        states=set(str(_) for _ in [*automat.states, 'black_hole']),
        input_symbols=symbols,
        transitions=transitions,
        initial_state='0',
        final_states=set(str(_) for _ in automat.terminals)
    )
    bb = aa.minify()
    print('all states:', len(bb.states), bb.states)
    print('final states:', len(bb.final_states), bb.final_states)
    return bb
