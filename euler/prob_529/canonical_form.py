import json

from euler.lib.timer import timer
from euler.prob_529.digit529 import Digit529


@timer
def canonical_all_ok(x: Digit529):
    y = x.mirror()
    for i in range(len(x.digits)):
        part = y.slice(last=i + 1)
        if part.full_check():
            return part.mirror()
    return x


@timer
def canonical_all_ko(x: Digit529):
    y = x.mirror()
    tot = 0
    for i in range(len(y.digits)):
        tot += y.digits[i]
        if tot >= x.problem.N:
            z = y.slice(last=i + 1).mirror()
            return z
    return x


@timer
def canonical_mixed(valid: Digit529, invalid: Digit529):
    if sum(invalid.digits) == valid.problem.N - 1:
        return invalid
    z = Digit529(valid.problem, canonical_all_ok(valid).digits + invalid.digits)
    return z


with open('../resources/p529/classes.json') as _:
    CLASSES = {
        int(k): list(sorted(map(int, l)))
        for k, l in json.load(_).items()
    }


@timer
def canonical_form(x: Digit529):
    if x.digits == [0]:
        return x

    x = x.clean([0])

    if sum(x.digits) <= x.problem.N:
        return x.clean([0])

    sign = x.sgn

    if sign[0] is True and sign[-1] is True:
        z = canonical_all_ok(x)
    elif sign[0] is False and sign[-1] is False:
        assert False  # never happen ... why ?
        z = canonical_all_ko(x)
    else:
        i = sign.index(False)
        valid, invalid = x.slice(last=i), x.slice(first=i)
        z = canonical_mixed(valid, invalid)
    return z
