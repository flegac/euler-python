from multiprocessing import Pool

from euler.lib.timer import show_timers, timer
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

    v = solver.compute_value(10 ** 18)
    print('-----------------------------------------')
    print('solve(10**18):', v)
    print('-----------------------------------------')

    # k2 = 1
    # k5 = 1
    # with open('../resources/p529/logs.txt', 'w') as _:
    #     pass
    # for i, x in enumerate(solver.enumerate(2 ** 18)):
    #
    #     if i in [111, 222, 444, 56, 55, 28, 27, 25,888,1776,3552]:
    #         v = solver.compute_value(i)
    #         assert v == x, '{}: mat_power:{} != mat_apply:{}'.format(i, v, x)
    #         print('{:5d}: {}'.format(i, x))
    #
    #     if i == 2 ** k2:
    #         print('computing 2**{}'.format(k2))
    #         v = solver.compute_value(i)
    #         assert v == x, '2**{}: mat_power:{} != mat_apply:{}'.format(k2, v, x)
    #         k2 += 1
    #         print('{:5d}: {}'.format(i, x))
    #
    #     if i == 5 ** k5:
    #         print('computing 5**{}'.format(k5))
    #         v = solver.compute_value(i)
    #         assert v == x, '5**{}: mat_power:{} != mat_apply:{}'.format(k5, v, x)
    #         k5 += 1
    #         print('{:5d}: {}'.format(i, x))
    #
    #     with open('../resources/p529/logs.txt', 'a') as _:
    #         _.write('{:5d}: {}\n'.format(i, x))
    #     print('{:5d}: {}'.format(i, x))


solver = P529Solver(use_mod=True)
def power(n: int):
    return solver.compute_value(n)


@timer
def solve_it():
    plan = [
        (2,),

        (4, 3), (7, 6), (14, 13), (28, 27),

        (56, 55), (112, 111), (223, 222), (445, 444), (889, 888), (1777, 1776),
        (3553, 3552), (7106, 7105), (14211, 14210), (28422, 28421), (56844, 56843), (113687, 113686),

        (227374, 227373), (454748, 454747), (909495, 909494), (1818990, 1818989), (3637979, 3637978),
        (7275958, 7275957), (14551916, 14551915), (29103831, 29103830), (58207661, 58207660),
        (116415322, 116415321), (232830644, 232830643), (465661288, 465661287), (931322575, 931322574),
        (1862645150, 1862645149), (3725290299, 3725290298), (7450580597, 7450580596), (14901161194, 14901161193),
        (29802322388, 29802322387), (59604644776, 59604644775), (119209289551, 119209289550),
        (238418579102, 238418579101), (476837158204, 476837158203), (953674316407, 953674316406),
        (1907348632813, 1907348632812),

        (3814697265625,), (7629394531250,), (15258789062500,), (30517578125000,), (61035156250000,),
        (122070312500000,), (244140625000000,), (488281250000000,), (976562500000000,), (1953125000000000,),
        (3906250000000000,), (7812500000000000,), (15625000000000000,), (31250000000000000,), (62500000000000000,),
        (125000000000000000,), (250000000000000000,), (500000000000000000,), (1000000000000000000,)
    ]
    for args in plan:
        print('***** computing:', args)
        pool = Pool(2)
        res = pool.map(power, args)
        print('***** result:', args, res)
        pool.close()
        pool.join()


if __name__ == '__main__':
    try:
        p529(5)
        # solve_it()
    finally:
        show_timers()
