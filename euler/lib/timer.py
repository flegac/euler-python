import datetime
import time
from collections import defaultdict

_times = defaultdict(lambda: [0, 0])


def show_timers():
    print('{:20}: {:9} : {}'.format('name', 'calls', 'time'))
    for k in sorted(_times, key=lambda x: -_times[x][0]):
        v = _times[k]
        print('{:20}: {:9d} : {}'.format(k.__name__, v[1], datetime.timedelta(seconds=v[0])))


def reset_timers():
    _times.clear()


def timer(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        dt = time.time() - start
        _times[fn][0] += dt
        _times[fn][1] += 1
        return res

    return wrapper

# def time_now(fn):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         res = fn(*args, **kwargs)
#         dt = time.time() - start
#         print('{}: {}'.format(fn.__name__, datetime.timedelta(seconds=dt)))
#         return res
#
#     return wrapper
