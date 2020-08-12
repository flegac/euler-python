from typing import Callable, Any


def fast_power(x, n: int, mult: Callable[[Any, Any], Any]):
    if n == 0:
        raise ValueError
    if n == 1:
        return x
    if n == 2:
        return mult(x, x)
    k = n // 2
    a = fast_power(x, k, mult)
    b = fast_power(x, n - k, mult)
    return mult(a, b)
