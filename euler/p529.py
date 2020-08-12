from euler.lib.timer import show_timers
from euler.prob_529.p529_solver import P529Solver
from euler.prob_529.problem_529 import P529


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

2 9
3 72
4 507
5 3492
6 23697
7 158940
8 1057941
9 7012665
10 46402069

[0, 0, 9, 72, 507, 3492, 23697, 158940, 1057941, 7012665, 46402069]

    '''

    solver = P529Solver()

    # values1 = [solver.compute_value(_) for _ in range(7)]
    # print(values1)

    # values2 = [solver.compute_value2(_) for _ in range(7)]
    # print(values2)

    for i, x in enumerate(solver.enumerate(10)):
        print(x)


if __name__ == '__main__':
    p529(5)
    show_timers()

"""
B(n) = A + ... + A^n

B(1) = A
B(n) = B(n / 2) * (I + A^(n / 2)) if n is even
B(n) = B(n / 2) * (I + A^(n / 2)) + A^n if n is odd
"""
