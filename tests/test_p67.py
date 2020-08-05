from euler.p67 import p67
from euler.p529 import p529


def test_p67():
    assert p67() == 7273


def test_p529():
    assert p529(2) == 9
    assert p529(5) == 3492
    assert p529(10 ** 18) == -1
