from euler.lib.divisors import divisors


def d(n: int):
    return sum(divisors(n)) - n


def amicable_numbers(limit: int):
    res = set()
    for a in range(2, limit):
        b = d(a)
        c = d(b)
        if a == c != b:
            res.add(a)
            res.add(b)
            print(a, b, c)
    return res


def p21(n: int):
    res = sum(amicable_numbers(n))
    print(res)
    return res


def p22(n: int):
    pass


if __name__ == '__main__':
    p21(10_000)
