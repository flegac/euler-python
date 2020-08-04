from euler.p1_10 import p1, p2, p3, p4, p5, p6, p7, p8, p9, p10


def test_p1():
    assert p1(1000) == 233168


def test_p2():
    assert p2(4_000_000) == 4613732


def test_p3():
    assert p3(13195) == 29
    assert p3(600851475143) == 6857


def test_p4():
    assert p4(2) == 9009
    assert p4(3) == 906609


def test_p5():
    assert p5(10) == 2520
    assert p5(20) == 232792560


def test_p6():
    assert p6(10) == 2640
    assert p6(100) == 25164150


def test_p7():
    assert p7(6) == 13
    assert p7(10_001) == 104743


def test_p8():
    assert p8(4) == 5832
    assert p8(13) == 23514624000


def test_p9():
    assert p9(1000) == 31875000


def test_p10():
    assert p10(10) == 17
    assert p10(2_000_000) == 142913828922
