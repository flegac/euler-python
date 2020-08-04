from math import ceil, sqrt
from typing import Callable

from multiset import Multiset


class Prime(object):
    def __init__(self):
        self.sorted_primes = [2]
        self.primes = set(self.sorted_primes)
        self.limit = 2

    def check(self, n: int):
        self.find_primes(lambda _: _.limit > n)
        return self._test(n)

    def factorize(self, n: int):
        res, n = self._find_divisors(n)
        while n != 1:
            self._find_next_prime()
            res2, n = self._find_divisors(n, first=-1)
            res += res2
        return res

    def _find_divisors(self, n: int, first: int = 0):
        assert n != 0
        res = Multiset()
        for p in self.sorted_primes[first:]:
            while n % p == 0:
                res.add(p)
                n //= p
        return res, n

    def _find_next_prime(self):
        while True:
            self.limit += 1
            if self._test(self.limit):
                self.primes.add(self.limit)
                self.sorted_primes.append(self.limit)
                return

    def find_primes(self, stop_condition: Callable[['Prime'], bool] = lambda _: _.limit > 1000):
        if self.limit % 2 == 0:
            self.limit += 1
        while not stop_condition(self):
            if self._test(self.limit):
                self.primes.add(self.limit)
                self.sorted_primes.append(self.limit)
            self.limit += 2

    def _test(self, n: int):
        root = ceil(sqrt(n))
        if n in self.primes:
            return True
        for _ in self.sorted_primes:
            if _ > root:
                return True
            if n % _ == 0:
                return False
        return True
