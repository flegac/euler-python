from euler.lib.timer import show_timers
from euler.prob_529.p529 import P529
from euler.prob_529.p529_solver import P529Solver


def L(n: int):
    problem = P529(10)
    for i in range(10 ** n):
        if problem.item(i).full_check():
            yield i


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
    2) Prove that:
        |L(n)| = sum(k=0..n) [ |L(k;0)| * binomial(n,k) ]
        |L(n;0)| = sum(u in B) sum(v in T(n)) [ M(n)[u][v] ]

[0, 0, 9, 72, 507, 3492, 23697, 158940, 1057941, 7012665, 46402069]
    '''

    solver = P529Solver(use_mod=True)

    # v = solver.compute_value(10 ** 18)
    # print('-----------------------------------------')
    # print('solve(10**18):', v)
    # print('-----------------------------------------')

    k = 0
    with open('../resources/p529/logs.txt', 'w') as _:
        pass
    for i, x in enumerate(solver.enumerate(512)):

        if True or i % (2 ** k) == 0:
            # v = solver.compute_value(i)
            # assert v == x, '{}: mat_power:{} != mat_apply:{}'.format(i, v, x)

            with open('../resources/p529/logs.txt', 'a') as _:
                _.write('{:5d}: {}\n'.format(i, x))
            print('{:5d}: {}'.format(i, x))
            k += 1


if __name__ == '__main__':
    p529(5)
    show_timers()
