class Collatz(object):
    def __init__(self):
        self.graph = dict()

    def compute(self, n0: int):
        if n0 == 1:
            self.graph[n0] = (None, 1)
        if n0 not in self.graph:
            n1 = Collatz.next(n0)
            self.graph[n0] = (n1, 1 + self.compute(n1)[1])
        return self.graph[n0]

    @staticmethod
    def next(n: int):
        if n == 1:
            return None
        if n % 2 == 0:
            return n // 2
        return 3 * n + 1