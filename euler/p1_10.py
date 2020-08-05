from functools import reduce
from itertools import product

import numpy as np
from multiset import Multiset

from euler.lib.fibo import fib
from euler.lib.multiple import multiple_of
from euler.lib.palindrom import palindrom
from euler.lib.prime import Prime


def p1(n: int):
    values = filter(multiple_of([3, 5]), range(n))
    x = sum(values)
    print(x)
    return x


def p2(n: int):
    x = sum(filter(lambda x: x % 2 == 0, fib(n)))
    print(x)
    return x


def p3(n: int):
    P = Prime()
    res = P.factorize(n)
    print(res)
    return max(res)


def p4(n: int):
    gen = list(range(10 ** (n - 1), 10 ** n))
    tmp = [x * y for x, y in product(gen, gen)]
    res = sorted(filter(palindrom, tmp))
    res = max(res)
    print(res)
    return res


def p5(n: int):
    # What is the smallest positive number
    # that is evenly divisible
    # by all of the numbers from 1 to n ?

    P = Prime()
    divisors = list(map(P.factorize, range(2, n)))

    x = Multiset()
    for _ in divisors:
        x = x.union(_)

    x = np.prod(x)
    print(x)
    return x


def p6(n: int):
    x1 = sum([x ** 2 for x in range(1, n + 1)])
    x2 = (n * (n + 1) // 2) ** 2
    res = x2 - x1
    print(x2, ' - ', x1, ' = ', res)
    return res


def p7(n: int):
    P = Prime()
    P.find_primes(lambda x: len(P.primes) >= n)

    print(P.sorted_primes[-10:])
    return P.sorted_primes[-1]


def p8(n: int):
    x = '''73167176531330624919225119674426574742355349194934
96983520312774506326239578318016984801869478851843
85861560789112949495459501737958331952853208805511
12540698747158523863050715693290963295227443043557
66896648950445244523161731856403098711121722383113
62229893423380308135336276614282806444486645238749
30358907296290491560440772390713810515859307960866
70172427121883998797908792274921901699720888093776
65727333001053367881220235421809751254540594752243
52584907711670556013604839586446706324415722155397
53697817977846174064955149290862569321978468622482
83972241375657056057490261407972968652414535100474
82166370484403199890008895243450658541227588666881
16427171479924442928230863465674813919123162824586
17866458359124566529476545682848912883142607690042
24219022671055626321111109370544217506941658960408
07198403850962455444362981230987879927244284909188
84580156166097919133875499200524063689912560717606
05886116467109405077541002256983155200055935729725
71636269561882670428252483600823257530420752963450'''
    x = ''.join(x.split('\n'))
    m = 0
    for i in range(1 + len(x) - n):
        s = x[i:i + n]
        digits = [int(_) for _ in s]
        value = reduce(lambda a, b: a * b, digits)
        if value > m:
            m = value
    print(m)
    return m


def p9(n: int):
    # find (a,b,c) such that:
    # a < b < c
    # c² = a² + b²
    # c = n - (a + b)

    for b in range(n):
        for a in range(b):
            c = n - (a + b)
            if a ** 2 + b ** 2 == c ** 2:
                print(a, b, c)
                print(a * b * c)
                return a * b * c


def p10(n: int):
    P = Prime()
    P.find_primes(lambda x: x.limit >= n)
    res = sum(P.primes)
    print(res)
    return res


if __name__ == '__main__':
    p10(10)
    p10(2_000_000)
