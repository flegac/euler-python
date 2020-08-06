from euler.lib.prime import Prime


def test_check():
    P = Prime()

    primes = [11, 31, 6857]

    not_primes = [22, 693]

    for p in primes:
        assert P.check(p)
    for p in not_primes:
        assert not P.check(p)


def test_factorize():
    P = Prime()
    y = P.factorize(600851475143)
    print(y)
