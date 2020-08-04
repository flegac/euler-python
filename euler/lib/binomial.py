def binomial(n: int, k: int):
    return fact(n) // fact(k) // fact(n - k)


def fact(n: int):
    if n == 1: return 1
    return n * fact(n - 1)
