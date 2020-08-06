from euler.p21_30 import p21, d


def test_p21():
    assert d(220) == 284
    assert d(284) == 220
    assert p21(10_000) == 31626
