from math import ceil, sqrt


def divisors(n: int):
    res = set()
    for i in range(1, ceil(sqrt(n))):
        if n % i == 0:
            res.update([i, n // i])
    return res