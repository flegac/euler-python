from itertools import product
from typing import Tuple, Set

from euler.lib.timer import show_timers
from euler.problem_529 import P529, mirror, digits


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
    test = P529(10).full_check

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

    g = problem.build_graph()

    V = list(map(mirror, sorted(map(mirror, g.vertices))))

    for i, v in enumerate(V):
        print('{:4d} {}: {}'.format(i, problem.full_check(digits(v)), v))
    print('vertex:', len(g.vertices))

    # pour déterminer si deux états sont identiques :
    # - utiliser la signature (le nombre de VRAI)
    # - utiliser la dernière partie de somme <= 10

    mat = g.matrix()

    # M = power(mat, 2, 1_000_000_7)
    # print(M)
    # tot = 0
    # for a in problem.A:
    #     for b in T:
    #         i = g.vertices.index(a)
    #         j = g.vertices.index(b)
    #         val = M[i, j]
    #         tot += val
    # print(tot)


if __name__ == '__main__':
    p529(5)

    show_timers()
