from typing import List

from multiset import Multiset


class Line:
    def __init__(self, word: List[int]):
        self.word = word

    def count_suits(self):
        pass

    def check(self):
        return False

class P696:
    """


    Pour la suite, on considère que la longueur des suites, n, est fixé.
    On note alors w(n,s,t) = w(s,t)

    Soit l'alphabet A = {0,...,4}.
    Soit S_i, pour i<s, l'ensemble des cartes d'une liste (une suite contient n cartes).

    Soit f(t) le nombre de main pseudo-gagnantes, composées de 3t+2 cartes, sur une seule suite.
    Soit g(t) le nombre de main gagnantes n'utilisant qu'une seule suite, f(t) = w(s=1,t).

    P1: Les contraintes chow/pung entre deux suites S_i et S_j sont indépendantes.

    Soit partition(t) l'ensemble des partitions de l'entier t, en s valeurs dans {0,...,t}.
    Pour tout p dans partition(t), p = {p_1, ..., p_s} avec p_i in {0,...,t} et sum(p_i) = t.
    Nb: partition(t) est l'ensemble des façons de partager t entre toutes les suites.
    Pour tout p in partition(t) on note |p| le nombre de valeurs non nulles dans p.

    w(s,t) = Sum((i, p) avec p in partition(t) et i < s) [ f(p_0) * ...* g(p_i) * ... * f(p_k) ]
    * i est l'indice de la suite contenant une pair.

    P2: Pour tout p dans partition(p),






    """

    def __init__(self, n: int, s: int, t: int):
        self.n = n
        self.s = s
        self.t = t

        self.figure_number = n * s
        self.figure_repetition = 4


class WinningHand:
    def __init__(self, pair: int, chows: List[int], pungs: List[int]):
        self.pair = pair
        self.chows = Multiset(chows)  # suite
        self.pungs = Multiset(pungs)  # triplet

    def check(self, problem: P696):
        # pair
        pair = self.pair
        assert 0 <= pair < problem.figure_number

        # triplets
        pungs = h.pungs
        assert 0 <= min(pungs.distinct_elements())
        assert max(pungs.distinct_elements()) < problem.figure_number
        assert max(pungs.multiplicities()) == 1
        assert pair not in self.pungs

        # suites
        chows = h.chows
        for e in chows.distinct_elements():
            assert e % problem.n < problem.n - 2

        # main globale
        assert len(self.chows) + len(self.pungs) == problem.t


        #TODO : check impossibilities
        for value, _ in pungs.items():
            count_chows = chows.get(value, 0)
            if value % problem.n > 0:
                count_chows += chows.get(value-1, 0)
            if value % problem.n < problem.n-1:
                count_chows += chows.get(value+1,0)
            assert count_chows <= 4

if __name__ == '__main__':
    p = P696(n=6, s=6, t=6)

    h = WinningHand(
        pair=2,
        chows=[3, 3, 3],
        pungs=[3, 5, 4])
    h.check(p)
