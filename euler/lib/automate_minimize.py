from typing import Set, List

from euler.lib.automate import Automate
from euler.lib.timer import timer


def freeze(x):
    return frozenset(x)


@timer
def minimize(aut: Automate):
    # TODO: optimize encoding of sets
    F = freeze(aut.T)
    Q = freeze(aut.S)
    Q_F = freeze(set(Q).difference(F))
    P = {F, Q_F}
    W = {
        (min([F, Q_F], key=len), a)
        for a in aut.A
    }

    while len(W) > 0:
        Z, a = W.pop()
        assert sum(map(len, P)) == len(aut.S)
        P2 = set()
        for X in P:
            res = split(aut, Z, a, X)
            for _ in res:
                P2.add(_)
            if len(res) == 2:
                smallest = min(res, key=len)
                for b in aut.A:
                    if (X, b) in W:
                        W.remove((X, b))
                        for _ in res:
                            W.add((_, b))
                    else:
                        W.add((smallest, b))

        P = P2

    partition = list(map(list, P))
    return partition


def split(aut: Automate, Z: Set[str], a: str, X: Set[str]):
    A = set()
    B = set()
    for s1 in X:
        img = aut.find(s1, a)
        if len(img) == 0:
            B.add(s1)
        else:
            for s2 in img:
                if s2 in Z:
                    A.add(s1)
                else:
                    B.add(s1)
    assert len(A) + len(B) == len(X)
    if len(A) == 0 or len(B) == 0:
        return [X]
    return [freeze(A), freeze(B)]


@timer
def update_classes(automate: Automate, classes: List[List[int]]):
    # init mapping function :
    mapping = dict()
    cc = set()
    for c in classes:
        c = list(sorted(c, key=lambda x: (len(x), x)))
        x = c[0]
        cc.add(x)
        for _ in c[1:]:
            mapping[_] = x

    def f(s):
        return mapping.get(s, s)

    # check all
    for s1, a, s2, n in automate.edges():
        items = automate.find(f(s1), a)

        if f(s2) not in set(map(f, items)):
            print('-----------------------------------------------------------')
            print('error on:', f(s1), a, f(s2), n)
            print('found   :', items)
            raise ValueError

    # update edges (s1,a,s2,n) --> (s1,a,f(s2),n)
    Q = Automate.from_scratch()
    for s1, a, s2, n in automate.edges():
        if not Q.find(f(s1), a):
            Q.add(f(s1), a, f(s2), n)
    Q.I = f(automate.I)
    Q.T = set(map(f, automate.T))
    Q.validate()
    return Q
