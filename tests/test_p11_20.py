from euler.lib.words import Words
from euler.p11_20 import p11, p12, p13, p14, p15, p16, p17, p18, p20


def test_p11():
    assert p11(4) == 70600674


def test_p12():
    assert p12(5) == 28
    assert p12(500) == 76576500


def test_p13():
    assert p13() == '5537376230'


def test_p14():
    assert p14(1_000_000) == 837799


def test_p15():
    assert p15(2) == 6
    assert p15(20) == 137846528820


def test_p16():
    assert p16(15) == 26
    assert p16(1000) == 1366


def test_p17():
    w = Words()
    assert w.words(342) == 'three hundred and forty-two'
    assert w.words(115) == 'one hundred and fifteen'

    assert p17(5) == 19
    assert p17(1000) == 21124


def test_p18():
    assert p18() == 1074


def test_p19():
    assert False


def test_p20():
    assert p20(10) == 27
    assert p20(100) == 648
