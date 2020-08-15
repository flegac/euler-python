from pathlib import Path
from typing import Callable, Any

import numpy as np


class FastPower(object):
    def __init__(self, x: np.ndarray, mult: Callable[[Any, Any], Any], path: str = None):
        self.x = x
        self.mult = mult
        self.cache = dict()
        self.path = path

    def full_path(self, n):
        return Path('{}/{}.dat'.format(self.path, n))

    def power(self, n: int):
        if n == 0:
            raise ValueError
        if n == 1:
            return self.x
        if n in self.cache:
            return self.cache[n]

        if self.path:
            path = self.full_path(n)
            if path.exists():
                res = np.load(path, allow_pickle=True).astype(np.uint64)
                self.cache[n] = res
                return res

        k = n // 2
        a = self.power(k)
        b = self.power(n - k)
        res = self.mult(a, b)
        self.cache[n] = res

        if self.path:
            res.astype(np.uint64).dump(self.full_path(n))

        return res
