from collections import defaultdict
from fractions import Fraction
from math import log2
from typing import Tuple, Set, Mapping

from euler.lib.timer import show_timers
from euler.prob_529.problem_529 import P529


def partition(number) -> Set[Tuple[int]]:
    answer = set()
    answer.add((number,))
    for x in range(1, number):
        for y in partition(number - x):
            answer.add(tuple(((x,) + y)))
    answer = sorted(answer)
    answer = sorted(answer, key=lambda x: len(x))
    return answer


def L(n: int, k: int = 1):
    problem = P529(10)

    A = list(range(k, 10))
    for i in range(10 ** n):
        if problem.item(i).full_check():
            yield i


M = 1_000_000_007


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
    - Let T(X) be the set of accepting states of X
    - Let M be the adjacency matrix of X
    - Let M(n) be the n-power of M


    1) Prove that:
        |L(n)| = sum(k=0..n) [ |L(n-k;0)| * binomial(n,k) ]
        |L(n;0)| = sum(u in B) sum(v in T(n)) [ M(n)[u][v] ]

     4) Full formulae is :
        |L(n)| = sum(k=0..n) sum(u in B) sum(v in T(n)) [ M(n-k)[u][v] * binomial(n,k) ]

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
    problem = P529(10)

    xx = initial_nodes()
    next_nodes, word_number = solver(problem)

    last = 0
    last_frac = -1
    print('length, nodes, value, ratio')

    n = 10 ** 18
    k = 1
    for i in range(2, n):
        xx = next_nodes(xx)
        current = last + word_number(xx)

        frac = Fraction(current, last) if last != 0 else 1
        if i % 2 ** k == 0 or i <= 10:
            print(i, len(xx), current, frac)
            k = 1 + int(log2(i))

        if frac == last_frac:
            break
        last = current
        last_frac = frac


def initial_nodes():
    xx = dict()
    for i in range(1, 10):
        xx[i] = 1
    return xx


def solver(problem: P529):
    g = problem.build_graph()
    terminal = list(filter(lambda _: problem.item(_).full_check(), g.vertices))
    print('vertices:', len(g.vertices), 'terminals:', len(terminal))

    def next_nodes(nodes: Mapping):
        res = defaultdict(lambda: 0)
        for v1, kk in nodes.items():
            for a, v2 in g.graph[v1]:
                res[v2] += kk
        return res

    def word_number(nodes: Mapping):
        tot = 0
        for v in terminal:
            tot += nodes.get(v, 0)
        return tot

    return next_nodes, word_number


if __name__ == '__main__':
    for i in range(2, 6):
        print(i, len(list(L(i))))

    p529(5)
    show_timers()
