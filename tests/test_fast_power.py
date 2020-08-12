from euler.lib.fast_power import fast_power


def test_fast_power():
    x = 13
    n = 111

    def mult(a, b):
        return a * b

    y = fast_power(x, n, mult)
    assert y == x ** n
