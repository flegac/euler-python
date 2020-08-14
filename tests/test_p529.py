from euler.p529 import digits, from_digits
from euler.prob_529.p529 import P529


def test_canonical():
    # assert Problem(10).canonical_form(261144) == 1144

    p = P529(3)
    x = [0, 1, 2, 10, 20, 11, 21, 111, 112, 211, 212, ]
    z = list(map(from_digits, map(p.canonical_form, map(digits, x))))
    y = [0, 1, 2, 1, 2, 11, 21, 111, 112, 211, 12]

    assert z == y


def test_p529():
    p = P529(10)
    assert False
