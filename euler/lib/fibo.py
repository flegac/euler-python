def fib(limit: int):
    res = [1, 2]
    while res[-1] < limit:
        res.append(res[-2] + res[-1])
    return res