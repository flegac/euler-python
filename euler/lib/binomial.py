def binomial(n: int, k: int):
    return fact(n) // fact(k) // fact(n - k)


def fact(n: int):
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res
